# Points and Paths Data Format Specification

This document describes the JSON format for storing 3D points and paths data. The format supports both individual points and collections of points organized into paths.

## Data Types

### Point

A Point represents a location in 3D space with optional metadata.

**Required Fields:**

- `x` (number): X coordinate
- `y` (number): Y coordinate
- `z` (number): Z coordinate

**Optional Fields:**
Note: additional optional fields may be added in the future.

- `size` (number): Point size, must be greater than 0 (default: 1.0)
- `name` (string | null): Custom name for the point. If null or omitted, will be auto-generated as "Point(x,y,z)"

Example:

```json
{
  "x": 1.0,
  "y": 2.0,
  "z": 3.0,
  "size": 1.5,
  "name": "Start Point"
}
```

### Path

A Path represents a sequence of points with styling metadata. The order of the points is important.

**Required Fields:**

- `points` (array): Array of Point objects (minimum 1 point)

**Optional Fields:**
Note: additional optional fields may be added in the future.

- `name` (string | null): Custom name for the path. If null or omitted, will be auto-generated as "Path-{N}pts"
- `color` (string): Hex color code (default: "#0000FF")
- `thickness` (number): Line thickness, must be greater than 0 (default: 1.0)

Example:

```json
{
  "points": [
    { "x": 1.0, "y": 2.0, "z": 3.0 },
    { "x": 4.0, "y": 5.0, "z": 6.0 }
  ],
  "name": "Sample Path",
  "color": "#FF0000",
  "thickness": 2.0
}
```

## File Format Options

The JSON file can be structured in three different ways:

### 1. Mixed Dictionary Format

```json
{
  "points": [
    { "x": 1.0, "y": 2.0, "z": 3.0 },
    { "x": 4.0, "y": 5.0, "z": 6.0 }
  ],
  "paths": [
    {
      "points": [
        { "x": 7.0, "y": 8.0, "z": 9.0 },
        { "x": 10.0, "y": 11.0, "z": 12.0 }
      ],
      "name": "Path1"
    }
  ]
}
```

### 2. Array Format

Can contain either all points or all paths:

```json
[
  { "x": 1.0, "y": 2.0, "z": 3.0 },
  { "x": 4.0, "y": 5.0, "z": 6.0 }
]
```

### 3. Single Path Format

A single path object:

```json
{
  "points": [
    { "x": 1.0, "y": 2.0, "z": 3.0 },
    { "x": 4.0, "y": 5.0, "z": 6.0 }
  ],
  "name": "Single Path"
}
```

## Validation Rules

1. Points must have all three coordinates (x, y, z)
2. Size and thickness must be greater than 0
3. Colors must be valid hex codes (#RRGGBB format)
4. Paths must contain at least one point
5. Additional properties not defined in the schema will be ignored

## Implementation Notes

- When names are omitted or null:
  - Points will be named "Point(x,y,z)" with coordinates rounded to 1 decimal
  - Paths will be named "Path-{N}pts" where N is the number of points
- The format supports both single objects and collections
- Arrays must contain either all points or all paths (no mixing)
