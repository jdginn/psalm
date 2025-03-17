# Simulation Summary Format

This document describes the JSON format for storing binaural analysis results including Initial Time Difference (ITD) measurements.

## Data Structure

The JSON file must be a dictionary with the following structure:

```json
{
  "status": string,
  "errors"?: [string],
  "results": {
    "ITD"?: number,
    "ITD_2"?: number,
    "avg_gain_5ms"?: number
  }
}
```

## Fields

### Top Level Fields

**Required Fields:**

- `status` (string): Current status of the analysis. Must be one of:
  - `"success"`: Analysis completed successfully
  - `"validation_error"`: Input validation failed
  - `"simulation_error"`: Error occurred during simulation

**Optional Fields:**

- `errors` (array): Array of error message strings. May be omitted if there are no errors.

### Results Fields

**Required Fields:**

**Optional Fields:**

- `ITD` (number): Primary Initial Time Delay measurement in milliseconds
- `ITD_2` (number): Secondary Initial Time Delay measurement in milliseconds. This figure measures ITD for an alternate, configurable RFZ. This RFZ may be a subset or superset of the primary RFZ.
- `avg_gain_5ms` (number): Average gain of reflections over 5ms window following ITD
- `listen_pos_dist` (number): Distance from the listening position from the front wall in meters

## Example

```json
{
  "status": "success",
  "results": {
    "ITD": 0.14
    "ITD_2": 0.20
    "avg_gain_5ms": -6.5
    "listen_pos_dist": 1.4
  }
}
```

Minimal successful example:

```json
{
  "status": "success"
}
```

Example with validation error:

```json
{
  "status": "validation_error",
  "errors": [
    "Invalid input: Sample rate must be greater than 0",
    "Missing required field: channel_count"
  ]
}
```
