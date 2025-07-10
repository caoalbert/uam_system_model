## 🧭 Description

- `network`: Base layer — defines schedule, vertiports, and matrices  
- `fleetop`: Optimization logic — builds on `network`  
- `fleetopsummary`: Reporting and analysis — depends on `network` and the output file from `fleetop`

Each layer builds on the one before it. Generate network -> run optimization -> analytics