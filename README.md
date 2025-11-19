![CI](https://github.com/caoalbert/uam_system_model/actions/workflows/ci.yaml/badge.svg)
![release](https://img.shields.io/github/v/release/caoalbert/uam_system_model?include_prereleases&cacheSeconds=3600)
![GitHub contributors](https://img.shields.io/github/contributors/caoalbert/uam_system_model?cacheSeconds=3600)


## 🧭 Description

- `network`: Base layer — defines schedule, vertiports, and matrices  
- `fleetop`: Optimization logic — builds on `network`  
- `fleetopsummary`: Reporting and analysis — depends on `network` and the output file from `fleetop`

Each layer builds on the one before it. Generate network -> run optimization -> analytics