import json


class CostAnalyzer:
    def __init__(self, path_to_params):
        with open(path_to_params, "r") as file:
            self.params = json.load(file)
        self.capex = self.params["capex"]
        self.opex = self.params["opex"]

    def analyze(self, df):
        self.multiplier = 365 / df.shape[0]

        self.total_capex = self._compute_capex(df)
        self.total_opex = self._compute_opex(df)
        self.revenue = self._compute_revenue(df)

        return self.total_capex, self.total_opex, self.revenue

    def _compute_capex(self, df):
        fleet_size = int(df["fleet_size"][0])
        self.fleet_size = fleet_size
        for size in df["fleet_size"]:
            if fleet_size != size:
                raise Exception(
                    "Daily operation in the input file have different fleet sizes in rows"
                )
        fleet_aquisition_cost = fleet_size * self.capex["cost_per_vehicle"]

        required_pads = {}
        land_area = {}
        for row in df["pads_at_vertiport"]:
            for key, value in row.items():
                # Keep the max value seen so far for each key
                if key not in required_pads:
                    required_pads[key] = value
                else:
                    required_pads[key] = max(required_pads[key], value)

        for key, value in required_pads.items():
            land_area[key] = (
                self.capex["land"]["land_area_beta_0"]
                + value * self.capex["land"]["land_area_beta_1"]
            )
        land_cost = []
        for key, value in land_area.items():
            land_cost.append(value * self.capex["land"]["land_value_per_sqft"][key])
        land_acquisition_cost = sum(land_cost)

        construction_cost = sum(self.capex["construction_cost"].values())

        return fleet_aquisition_cost + construction_cost + land_acquisition_cost

    def _compute_opex(self, df):

        energy_cost = (
            self.opex["energy_cost_per_kWh"] * df["energy_consumption_kWh"]
        ).sum()

        pilot_cost = (
            self.opex["pilot_cost_per_aircraft_hour"] * df["number_of_aircraft_hours"]
        ).sum()

        battery_replacement_cost = (
            self.opex["battery_replacement_cost_per_aircraft_hour"]
            * df["number_of_aircraft_hours"]
        ).sum()

        maintenance_cost = (self.opex["maintenance_cost_per_asm"] * df["TAM"] * 4).sum()

        insurance_cost = (
            self.fleet_size
            * self.capex["cost_per_vehicle"]
            * self.opex["insurance_cost_factor"]
        )

        vertiport_operation_cost = self.opex["vertiport_operation_cost_per_year"] * len(
            df["pads_at_vertiport"][0]
        )

        net_opex = (
            (energy_cost + pilot_cost + battery_replacement_cost + maintenance_cost)
            * self.multiplier
            + insurance_cost
            + vertiport_operation_cost
        )
        total_opex = net_opex / (1 - self.opex["administrative_cost_factor"])

        return total_opex

    def _compute_revenue(self, df):
        revenue = df["total_revenue"].sum()
        return revenue * self.multiplier
