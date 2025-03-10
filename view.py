import sys
import argparse
import json
import typing
import random
import numpy as np
import trimesh
from models import Point, Path, AcousticPath, Zone

# Qt imports
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSplitter,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QColor

# OpenGL imports
import OpenGL.GL as gl


def load_json_data(file_path: str) -> dict:
    """Load and parse JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_point_cloud(points: list[Point]) -> typing.Union[trimesh.PointCloud, None]:
    """Create trimesh PointCloud from list of Points."""
    if not points:
        return None
    coords = [p.to_array() for p in points]
    return trimesh.PointCloud(coords)


def create_path_geometry(path: typing.Union[Path, AcousticPath]) -> trimesh.path.Path3D:
    """Create trimesh Path3D from Path or AcousticPath."""
    vertices = [p.to_array() for p in path.points]
    if isinstance(path, AcousticPath):
        vertices.append(path.nearest_approach.position.to_array())

    # If no color specified, generate a random one
    path_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Convert hex color to RGBA with 50% transparency
    rgba = list(trimesh.visual.color.hex_to_rgba(path_color))
    rgba[3] = 127  # Set alpha to 127 for 50% transparency

    # Create a single entity for the whole path
    entities = [trimesh.path.entities.Line(points=np.arange(len(vertices)), color=rgba)]

    return trimesh.path.Path3D(entities=entities, vertices=vertices)


def create_zone_geometry(zone: Zone) -> trimesh.Trimesh:
    """Create trimesh Trimesh from Zone."""
    # Create a unit sphere and scale/translate it
    sphere = trimesh.primitives.Sphere(radius=1.0)

    # Scale by radius
    sphere.apply_scale(zone.radius)

    # Translate to center position
    translation = np.array([zone.x, zone.y, zone.z])
    sphere.apply_translation(translation)

    # If no color specified, generate a random one
    if not hasattr(zone, "color") or zone.color is None:
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    else:
        random_color = zone.color

    # Get transparency value (default to 0.8 or 80%)
    transparency = getattr(zone, "transparency", 0.8)

    # Convert hex color to RGBA with specified transparency
    rgba = list(trimesh.visual.color.hex_to_rgba(random_color))
    rgba[3] = int((1 - transparency) * 255)  # Convert transparency to alpha

    # Apply the color to the sphere
    sphere.visual.face_colors = rgba

    return sphere


import numpy as np
from PyQt6.QtCore import Qt, QSize
from trimesh.viewer.trackball import Trackball  # Add this import
from OpenGL.GLU import gluPerspective  # Add this import


class TrimeshViewerWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(640, 480))

        # Camera and scene settings
        self.scene = None
        self.camera = trimesh.scene.Camera(fov=(60.0, 45.0), resolution=(800, 600))

        # Initialize view parameters
        self._initial_camera_transform = np.eye(4)
        self.view = {
            "ball": Trackball(
                pose=self._initial_camera_transform,
                size=(800, 600),
                scale=1.0,
                target=[0, 0, 0],
            )
        }

        # Mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.last_point = None
        self.rotating = False

        # Add zoom scale
        self._zoom_scale = 1.0

    def initializeGL(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background

    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        aspect = width / float(height)
        gluPerspective(45.0, aspect, 0.1, 1000.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)

        # Update trackball size
        self.view["ball"].resize((width, height))

    def mousePressEvent(self, event):
        self.last_point = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotating = True
            self.view["ball"].down(np.array([event.pos().x(), event.pos().y()]))

    def mouseReleaseEvent(self, event):
        self.rotating = False
        self.last_point = None

    def mouseMoveEvent(self, event):
        if self.rotating:
            self.view["ball"].drag(np.array([event.pos().x(), event.pos().y()]))
            self.update()

    def wheelEvent(self, event):
        # Scale the view based on the wheel delta
        delta = event.angleDelta().y() / 120.0
        self._zoom_scale *= 0.9 if delta < 0 else 1.1
        self.update()

    def paintGL(self):
        if not self.scene:
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Apply zoom scale
        gl.glScalef(self._zoom_scale, self._zoom_scale, self._zoom_scale)

        # Apply camera transform from trackball
        camera_transform = np.linalg.inv(self.view["ball"].pose)
        gl.glMultMatrixf(camera_transform.T.ravel())

        # Enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Draw all geometries in the scene
        for name, geometry in self.scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                # Draw mesh as wireframe
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glColor3f(1.0, 1.0, 1.0)  # White wireframe
                gl.glBegin(gl.GL_TRIANGLES)
                for face in geometry.faces:
                    for vertex in face:
                        gl.glVertex3fv(geometry.vertices[vertex])
                gl.glEnd()

            elif isinstance(geometry, trimesh.path.Path3D):
                # Draw paths
                gl.glLineWidth(2.0)  # Make lines thicker
                gl.glBegin(gl.GL_LINE_STRIP)

                # Get color from entity if available, otherwise use default
                if geometry.entities and hasattr(geometry.entities[0], "color"):
                    color = geometry.entities[0].color
                    gl.glColor4f(
                        color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255
                    )
                else:
                    gl.glColor4f(1.0, 0.0, 0.0, 0.5)  # Default: semi-transparent red

                for vertex in geometry.vertices:
                    gl.glVertex3fv(vertex)
                gl.glEnd()

            elif isinstance(geometry, trimesh.points.PointCloud):
                # Draw points (like nearest approach points)
                gl.glPointSize(5.0)
                gl.glBegin(gl.GL_POINTS)
                gl.glColor3f(1.0, 1.0, 0.0)  # Yellow points
                for vertex in geometry.vertices:
                    gl.glVertex3fv(vertex)
                gl.glEnd()

        gl.glDisable(gl.GL_BLEND)

    def reset_view(self):
        if self.scene is not None:
            self.view["ball"].pose = self._initial_camera_transform
            self._zoom_scale = 1.0  # Reset zoom
            self.update()


class AcousticPathViewer(QMainWindow):
    def __init__(self, room_mesh, acoustic_paths, points=None, paths=None, zones=None):
        super().__init__()
        self.setWindowTitle("Acoustic Path Viewer")

        # Store data
        self.room_mesh = room_mesh
        self.acoustic_paths = acoustic_paths
        self.points = points or []
        self.paths = paths or []
        self.zones = zones or []
        self.current_index = 0

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Info display
        self.info_label = QLabel("Path Information")
        left_layout.addWidget(self.info_label)

        # Navigation controls
        nav_frame = QFrame()
        nav_layout = QHBoxLayout(nav_frame)

        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.show_previous_path)
        self.next_button.clicked.connect(self.show_next_path)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        left_layout.addWidget(nav_frame)

        # Reset view button
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        left_layout.addWidget(self.reset_button)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # Add panels to splitter
        splitter.addWidget(left_panel)

        # Create and add viewer widget
        self.viewer_widget = TrimeshViewerWidget()
        splitter.addWidget(self.viewer_widget)

        # Set splitter proportions
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # Initial display
        self.update_display()
        self.update_info()

    def show_previous_path(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            self.update_info()

    def show_next_path(self):
        if self.current_index < len(self.acoustic_paths) - 1:
            self.current_index += 1
            self.update_display()
            self.update_info()

    def reset_view(self):
        self.viewer_widget.rotation = [30, 45, 0]
        self.viewer_widget.zoom = -5.0
        self.viewer_widget.translation = [0, 0, 0]
        self.viewer_widget.update()

    def update_info(self):
        current_path = self.acoustic_paths[self.current_index]
        info_text = (
            f"Path {self.current_index + 1} of {len(self.acoustic_paths)}\n"
            f"Points: {len(current_path.points)}\n"
            f"Nearest approach: {current_path.nearest_approach.position}"
        )
        self.info_label.setText(info_text)

    def update_display(self):
        scene = trimesh.Scene()
        scene.add_geometry(self.room_mesh, geom_name="room")

        current_path = self.acoustic_paths[self.current_index]
        scene.add_geometry(create_path_geometry(current_path), geom_name="current_path")
        scene.add_geometry(
            trimesh.PointCloud([current_path.nearest_approach.position.to_array()]),
            geom_name="nearest_approach",
        )

        self.viewer_widget.scene = scene
        self.viewer_widget.update()


def main():
    # Set up OpenGL format
    format = QSurfaceFormat()
    format.setDepthBufferSize(24)
    format.setStencilBufferSize(8)
    format.setSamples(4)  # Enable antialiasing
    format.setVersion(2, 1)
    QSurfaceFormat.setDefaultFormat(format)

    # Create application
    app = QApplication(sys.argv)

    # Parse arguments and load data
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument(
        "--annotations", help="Annotations file (optional)", default=None
    )
    args = parser.parse_args()

    # Load mesh and data
    room_mesh = trimesh.load(args.mesh)
    room_mesh.fix_normals()

    points = []
    paths = []
    acoustic_paths = []
    zones = []

    if args.annotations:
        data = load_json_data(args.annotations)
        points = [Point.from_dict(p) for p in data.get("points", [])]
        paths = [Path.from_dict(p) for p in data.get("paths", [])]
        acoustic_paths = [
            AcousticPath.from_dict(p) for p in data.get("acousticPaths", [])
        ]
        zones = [Zone.from_dict(p) for p in data.get("zones", [])]

    # Create and show window
    window = AcousticPathViewer(room_mesh, acoustic_paths, points, paths, zones)
    window.resize(1200, 800)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
