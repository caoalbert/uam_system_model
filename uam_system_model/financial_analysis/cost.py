import pandas as pd
import ast
import math
from collections import defaultdict

class CostAnalyzer:
    def __init__(self, energy_cost=3, maintenance_cost=3, overhead_cost=4, insurance_cost=5,
                 traffic_mangement_cost=2, pilot_cost=0.2,
                 cost_per_vehicle = 3e3, infra_vertiport_cost=1e6, infra_pad_cost=1e3, infra_tlof_cost = 1e3,
                 max_takeoff_per_hour_per_pad = 30, surface_area_param1 = 10/2.4, surface_area_param2 = 1/2.4,
                 land_cost: dict = None
                 ):
        """
        All cost parameters should be defined in 1000 dollars
        """
        # OPEX input parameters
        self.ec = energy_cost # per kWh
        self.mc = maintenance_cost # per flight miles TODO: figure out if battery is part of this number
        self.oc = overhead_cost # TODO: constant for now - multiplier based on total OPEX on average of airliners
        self.ic = insurance_cost # per vehicle per day # TODO: see if we can check
        self.tmc = traffic_mangement_cost # TODO: constant for now (per flight or available seat-miles or AC hours
        # TODO: Battery cost - considering SOH / cycle analysis
        self.pc = pilot_cost # per flight hours

        # CAPEX input parameters
        self.ivc = infra_vertiport_cost
        self.ipc = infra_pad_cost
        self.itc = infra_tlof_cost
        self.cpv = cost_per_vehicle

        # data
        self.input_df = pd.DataFrame()

        # inferred parameters
        self.vertiport_specification = {}
        self.fleet_size = 0
        self.param_mto_per_pad = max_takeoff_per_hour_per_pad

        # linear regression based surface area calculation
        # surface area = (pads + 10 / 2.4) [10000 sq. ft] - scaled based on Emin's dissertation
        self.const_sa1 = surface_area_param1
        self.const_sa2 = surface_area_param2

        # land cost
        self.land_cost = land_cost


    def load_data(self, data_dir):
        self.input_df = pd.read_csv(data_dir)

        self.input_df["pads_at_vertiport"] = self.input_df["pads_at_vertiport"].apply(ast.literal_eval)
        self.input_df["max_takeoff_per_hour"] = self.input_df["max_takeoff_per_hour"].apply(ast.literal_eval)

        self.vertiport_specification = self.compute_inferred_infrastructure_config(self.input_df)

        if self.land_cost is None:
            self.land_cost = {'LAX': 1000,
                              'DTLA': 1000,
                              'LGB': 1000,
                              'WDHL': 1000,
                              'UVS': 1000,
                              'ANH': 1000,
                              'HWD': 1000,
                              'PSD': 1000,
                              'BVH': 1000}

    def compute_inferred_infrastructure_config(self, input_df):
        """
        :param input_df: output from optimization
        :return: dictionary containing required pads and tlofs
        """
        fleet_size = int(input_df["fleet_size"][0])
        required_pads = defaultdict(int)
        required_tlofs = defaultdict(int)
        required_surface_area_dict = dict()

        for size in input_df["fleet_size"]:
            if fleet_size != size:
                raise Exception("Daily operation in the input file have different fleet sizes in rows")

        for row_dict in input_df["pads_at_vertiport"]:
            for key, value in row_dict.items():
                # Keep the max value seen so far for each key
                required_pads[key] = max(required_pads[key], value)

        for row_dict in input_df["max_takeoff_per_hour"]:
            for key, value in row_dict.items():
                # Keep the max value seen so far for each key
                required_tlofs[key] = max(required_tlofs[key], value)

        required_pads_dict = dict(required_pads)
        required_tlofs_dict = dict(required_tlofs)

        for key, value in required_tlofs_dict.items():
            required_tlofs_dict[key] = math.ceil(value/self.param_mto_per_pad)

        # surface area calculation
        for key, value in required_pads_dict.items():
            required_surface_area_dict[key] = round(self.const_sa1 + value*self.const_sa2, 4)

        return {'fleet_size': fleet_size, 'pads': required_pads_dict, 'tlofs': required_tlofs_dict, 'land': required_surface_area_dict}

    def compute_capex(self):
        """
        Compute CAPEX
        :return:
        """

        # define cost vector
        cost_fleet_acquisition = self.vertiport_specification["fleet_size"]*self.cpv
        cost_construction = 0
        cost_land_acquisition = 0

        # construction cost
        for key, value in self.vertiport_specification['pads'].items():
            cost_construction += self.vertiport_specification['tlofs'][key]*self.itc # tlof cost
            cost_construction += value*self.ipc
            """TODO: add constant construction cost for building construction"""

        # land acquisition
        for key, value in self.vertiport_specification['land'].items():
            cost_land_acquisition += value*self.land_cost[key]

        return cost_fleet_acquisition, cost_construction, cost_land_acquisition

    def compute_opex(self, multiplier):
        """
        Compute annual OPEX
        :param multiplier: number measured period in a year (e.g. if 30 days, multiplier = 12)
        :return: annual OPEX and Revenue stream
        """
        df = self.input_df.copy()

        energy_cost = self.ec * df["energy_consumption_kWh"] # energy cost
        maintenance_cost = self.mc * df["TAM"] # maintenance cost
        overhead_cost = pd.Series(self.oc, index=df.index) # overhead
        insurance_cost = pd.Series(self.vertiport_specification["fleet_size"]*self.ic, index=df.index)
        traffic_management_cost = pd.Series(self.tmc, index=df.index)
        pilot_cost = self.pc * df["TAM"]

        opex_df = pd.concat([
            energy_cost.rename("energy_cost"),
            maintenance_cost.rename("maintenance_cost"),
            overhead_cost.rename("overhead_cost"),
            insurance_cost.rename("insurance_cost"),
            traffic_management_cost.rename("traffic_management_cost"),
            pilot_cost.rename("pilot_cost")
        ], axis=1)

        columns = [
            "energy_cost",
            "maintenance_cost",
            "overhead_cost",
            "insurance_cost",
            "traffic_management_cost",
            "pilot_cost",
        ]

        opex_df["daily_opex"] = opex_df[columns].sum(axis=1)

        monthly_opex = opex_df["daily_opex"].sum()
        annual_opex = monthly_opex*multiplier

        return opex_df, monthly_opex, annual_opex
#
# def __main__():
#     data_dir = "./uam_system_model/data/sample_financial_input.csv"
