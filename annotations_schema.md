# Points and Paths Data Format Specification

This document describes the JSON format for storing 3D points and paths data. The format supports both individual points and collections of points organized into paths. Some paths include additional acoustic information.

## Data Types

### Point

A point in 3D space with optional metadata.

**Required Fields:**

- `x` (number): X coordinate
- `y` (number): Y coordinate
- `z` (number): Z coordinate

**Optional Fields:**

- `size` (number): Point size, must be greater than 0 (default: 1.0)
- `color` (string): Hex color code (default: "#000000")

### Vector

A 3D vector representing a direction.

**Required Fields:**

- `x` (number): X component
- `y` (number): Y component
- `z` (number): Z component

### Ray

A ray defined by an origin point and direction vector.

**Required Fields:**

- `origin` (Point): Starting point of the ray
- `direction` (Vector): Direction vector of the ray (should be normalized)

### Shot

Information about an acoustic ray shot.

**Required Fields:**

- `ray` (Ray): The ray that was shot
- `gain` (number): Initial gain of the shot in dB

### Path

A sequence of points with optional styling metadata.

**Required Fields:**

- `points` (array): Array of Point objects (minimum 1 point)

**Optional Fields:**

- `name` (string | null): Custom name for the path
- `color` (string): Hex color code (default: "#0000FF")
- `thickness` (number): Line thickness (default: 1.0)

### Material

Information about a material that can absorb or reflect sound.

**Required Fields:**

- `absorption` (float): Absorption coefficient of the material (should be between 0 and 1.0) where 0 is no absorption and 1.0 is complete absorption.

### Surface

Information about a surface that can absorb or reflect sound.

**Required Fields:**

- `material` (Material): Material of this surface

**Optional Fields:**

- `name` (string | null): Name of thes urface

### Reflection

Information about an acoustic reflection.

**Required Fields:**

- `position` (Point): Position of the reflection

**Optional Fields:**

- `normal` (Vector): Normal direction at the reflection
- `surface` (Surface): Surface that the reflection occurred on

### AcousticPath

A special type of path representing an acoustic reflection path.

**Required Fields:**

- `reflections` (array): Array of Reflection objects, starting with the shot origin (minimum 1 reflection)
- `shot` (Shot): The shot that created this path
- `gain` (number): Total gain in dB relative to direct signal
- `distance` (number): Total distance traveled in meters
- `nearestApproach` (NearestApproach): Information about closest point to listener

**Optional Fields:**

- `name` (string | null): Custom name for the path
- `color` (string): Hex color code (default: "#FF0000")
- `thickness` (number): Line thickness (default: 1.0)

### NearestApproach

Information about the point where the path comes closest to the listening position.

**Required Fields:**

- `position` (Point): Position of nearest approach
- `distance` (number): Distance to listening position at nearest approach

## Example

```json
{
  "points": [
    { "x": 0.0, "y": 0.0, "z": 1.5 }, // Shot origin
    { "x": 2.0, "y": 3.0, "z": 1.5 }, // First reflection
    { "x": 4.0, "y": 3.0, "z": 1.5 } // Second reflection
  ],
  "shot": {
    "ray": {
      "origin": { "x": 0.0, "y": 0.0, "z": 1.5 },
      "direction": { "x": 0.707, "y": 0.707, "z": 0.0 }
    },
    "gain": -3.0
  },
  "gain": -12.0,
  "distance": 5.2,
  "nearestApproach": {
    "position": { "x": 3.0, "y": 3.0, "z": 1.5 },
    "distance": 0.8
  },
  "color": "#FF0000",
  "thickness": 0.5
}
```

### Zone

A spherical region in 3D space.

**Required Fields:**

- `x` (number): X coordinate of center
- `y` (number): Y coordinate of center
- `z` (number): Z coordinate of center
- `radius` (number): Radius of the sphere in meters

**Optional Fields:**

- `name` (string | null): Custom name for the zone
- `color` (string): Hex color code (default: randomly generated)
- `transparency` (number): Transparency value between 0 and 1 (default: 0.8)

## File Format

The JSON file must be a dictionary with the following structure:

```json
{
  "points": [Point],
  "paths": [Path],
  "acousticPaths": [AcousticPath],
  "zones": [Zone]
}
```

All fields are optional - if a category has no entries, it can either be omitted or set to an empty array.

Example:

```json
{
  "points": [
    {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "name": "Origin",
      "size": 1.0,
      "color": "#000000"
    }
  ],
  "paths": [
    {
      "points": [
        { "x": 0.0, "y": 0.0, "z": 0.0 },
        { "x": 1.0, "y": 1.0, "z": 1.0 }
      ],
      "name": "Simple Path",
      "color": "#0000FF",
      "thickness": 1.0
    }
  ],
  "acousticPaths": [
    {
      "reflections": [
        {
          "position": { "x": 0.0, "y": 0.0, "z": 1.5 },
          "normal": { "x": 0.0, "y": 0.0, "z": 1.0 }
        },
        {
          "position": { "x": 2.0, "y": 3.0, "z": 1.5 },
          "normal": { "x": -0.707, "y": 0.707, "z": 0.0 },
          "surface": {
            "material": {
              "absorption": 0.3
            },
            "name": "Wall Surface"
          }
        },
        {
          "position": { "x": 4.0, "y": 3.0, "z": 1.5 },
          "normal": { "x": 0.0, "y": -1.0, "z": 0.0 },
          "surface": {
            "material": {
              "absorption": 0.5
            }
          }
        }
      ],
      "shot": {
        "ray": {
          "origin": { "x": 0.0, "y": 0.0, "z": 1.5 },
          "direction": { "x": 0.707, "y": 0.707, "z": 0.0 }
        },
        "gain": -3.0
      },
      "gain": -12.0,
      "distance": 5.2,
      "nearestApproach": {
        "position": { "x": 3.0, "y": 3.0, "z": 1.5 },
        "distance": 0.8
      },
      "name": "Example Acoustic Path",
      "color": "#FF0000",
      "thickness": 0.5
    }
  ],
  "zones": [
    {
      "x": 1.0,
      "y": 2.0,
      "z": 1.5,
      "radius": 0.5,
      "name": "Example Zone",
      "color": "#00FF00",
      "transparency": 0.8
    }
  ]
}
```
