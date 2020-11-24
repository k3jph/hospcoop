#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import argparse
import hospcoop
import math
import os
import pandas as pd
import pkg_resources
import pyomo
import pyomo.environ as pyo 
import random
import simpy
import spdlog
import sys
import time
import toml

from numpy.random import exponential, weibull, seed

infoBanner = "Hospital Cooperation During Acute Resource Shortages"
appName = os.path.basename(__file__)

CensusBlocks = []
Hospitals = []
Distance = { }
arrivalList = []
dischargeList = []
patientData = {}
Beds = {}
OccupiedBeds = {}
opt = None
solverType = None
solverVerbose = None
sched = None
MEANICUTIME = None

console = spdlog.ConsoleLogger(appName)
logLevelMap = {"trace" : spdlog.LogLevel.TRACE, "debug" : spdlog.LogLevel.DEBUG, "info" : spdlog.LogLevel.INFO}

env = simpy.Environment()


class Hospital(object):
    def __init__(self, env, key, name, beds):
        self.env = env
        self.id = key
        self.name = name
        self.icubeds = simpy.Resource(env, beds)

    def treat(self, patient, icutime):
        yield self.env.timeout(icutime)
        OccupiedBeds[self.id] = OccupiedBeds[self.id] - 1


def patient(env, idx, name, hosp):
    icutime = weibull(1) * MEANICUTIME
    patientData[idx]["hospitalArrivalTime"] = env.now[0]
    patientData[idx]["icuTime"] = icutime

    console.debug('%s arrives at the hospital %s at %.2f' % (name, hosp.name, env.now))
    with hosp.icubeds.request() as request:
        yield request
        patientData[idx]["icuStartTime"] = env.now[0]
        console.debug('%s enters the ICU at %.2f' % (name, env.now))

        yield env.process(hosp.treat(name, icutime))
        patientData[idx]["icuEndTime"] = env.now[0]
        console.debug('%s leaves the ICU at %.2f' % (name, env.now))
        dischargeList.append({"time": math.floor(env.now), "hospital": hosp.id})

def setup(env, patientGenerationRate, hospitalsData, tractsData):
    global CensusBlocks, Hospitals, Distance, Beds, OccupiedBeds, opt, solverType, solverVerbose, sched

    console.trace("Entering the setup function, creating the environment")
    hospitals = dict()
    for key in hospitalsData.keys():
        icus = int(hospitalsData[key]['icucount'])
        if(icus > 0):
            console.trace("Creating hospital %s with name %s and %s beds" % (key, hospitalsData[key]['name'], icus))
            hospitals[key] = Hospital(env, key, hospitalsData[key]['name'], icus)

    console.trace("Create tract generating function")
    for k in tractsData.keys():
        tractIDs = k 
        tractPops = tractsData.get(k)
    
    tractIDs = []
    tractPops = []
    for k in tractsData.keys():
        tractIDs.append(k)
        tractPops.append(int(tractsData[k]['population']))

    # Create more patients while the simulation is running
    i = 0
    while True:
        yield env.timeout(exponential(patientGenerationRate, 1))
        i += 1
        assignedHospital = 0
        tract = random.choices(population = tractIDs, weights = tractPops, k = 1)

        console.trace(f"Using minimum distance to assign hospital for patient {i}")
        minDistance = 1000
        for j in tractsData[tract[0]]["distance"].keys():
            if(tractsData[tract[0]]["distance"][j] < minDistance):
                assignedHospitalMin = j
                minDistance = tractsData[tract[0]]["distance"][j]
        console.debug(f"Minimum distance assigns patient {i} to hospital {assignedHospitalMin}")
        
        Location = tract[0]
        if sched == "lp":
            console.trace(f"Using LP to assign hospital for patient {i}")
            lpModel = build_lp_model(CensusBlocks, Hospitals, Distance, Beds, OccupiedBeds, arrivalList, dischargeList, Location)
            results = opt.solve(lpModel)
            for j in Hospitals:
                if lpModel.x[j] == 1:
                    assignedHospital = j
            console.debug(f"LP assigns patient {i} to hospital {assignedHospital}")
        elif sched == "nearest":
            console.trace(f"Using nearest-available bed to assign hospital for patient {i}")
            minDistance = 1000
            for j in tractsData[tract[0]]["distance"].keys():
                if tractsData[tract[0]]["distance"][j] < minDistance and OccupiedBeds[j] < Beds[j]:
                    assignedHospital = j
                    minDistance = tractsData[tract[0]]["distance"][j]
            console.debug(f"Nearest-available bed assigns patient {i} to hospital {assignedHospital}")
        else:
            assignedHospital = assignedHospitalMin

        OccupiedBeds[assignedHospital] = OccupiedBeds[assignedHospital] + 1
        patientData[i] = {}
        patientData[i]["homeTract"] = tract[0]
        patientData[i]["assignedHospital"] = assignedHospital
        patientData[i]["nearestHospital"] = assignedHospitalMin
        patientData[i]["distanceTraveled"] = tractsData[tract[0]]["distance"][assignedHospital]
        arrivalList.append({"time": math.floor(env.now), "hospital": j})

        if assignedHospital != assignedHospitalMin:
            console.warn(f"Assigned hospital for patient {i} is {assignedHospital}, but {assignedHospitalMin} is closer")
        else:
            console.info(f"Assigned hospital for patient {i} is {assignedHospital}")
        env.process(patient(env, i, 'Patient %d' % i, hospitals[assignedHospital]))


def build_lp_model(CensusBlocks, Hospitals, Distance, Beds, OccupiedBeds, arrivalList, dischargeList, Location):
    global MEANICUTIME

    model = pyo.ConcreteModel()
    ArrivalsDaily = {}
    Arrivals = {}
    DischargesDaily = {}
    Discharges = {}
    Timeperiods = range(MEANICUTIME)

    console.trace(f"Entering the LP model building at time {env.now[0]} for location {Location}")
    ## Okay, I am in a hurry and I will do it the long way...
    todayNumber = math.floor(env.now[0])
    for j in Hospitals:
        ArrivalsDaily[(j, 0)] = 0
        DischargesDaily[(j, 0)] = 0
        for t in range(todayNumber + 1):
            ArrivalsDaily[(j, t)] = 0
            DischargesDaily[(j, t)] = 0

    for p in arrivalList :
        dayNumber = math.floor(p["time"])
        ArrivalsDaily[(p["hospital"], dayNumber)] = ArrivalsDaily[(p["hospital"], dayNumber)] + 1

    for p in dischargeList:
        dayNumber = math.floor(p["time"])
        DischargesDaily[(p["hospital"], dayNumber)] = DischargesDaily[(p["hospital"], dayNumber)] + 1

    lastDayNumber = todayNumber - 1
    if lastDayNumber < 0:
        lastDayNumber = 0
    zeroDayNumber = lastDayNumber - MEANICUTIME
    if zeroDayNumber < 0:
        zeroDayNumber = 0

    console.trace(f"Finding mean Arrivals and Departures from day {zeroDayNumber} to day {lastDayNumber}")
    for j in Hospitals:
        tmpArrivalsSum = 0
        tmpDeparturesSum = 0
        for t in range(zeroDayNumber, lastDayNumber):
            tmpArrivalsSum = tmpArrivalsSum + ArrivalsDaily[(j, t)]
            tmpDeparturesSum = tmpDeparturesSum + DischargesDaily[(j, t)]
        ArrivalsMean = tmpArrivalsSum / MEANICUTIME
        DeparturesMean = tmpDeparturesSum / MEANICUTIME
        for t in range(MEANICUTIME):
            Arrivals[(j, t)] = ArrivalsMean
            Discharges[(j, t)] = DeparturesMean
    
    ## Variables 
    model.x = pyo.Var(Hospitals, within = pyo.Binary)
    model.penalty = pyo.Var(Hospitals, Timeperiods, domain = pyo.NonNegativeReals)

    ## Objective Fxn
    def obj_rule(model):
        return sum(Distance[Location, j] * model.x[j] for j in Hospitals) + sum(model.penalty[j, t] / (t + 1) for j in Hospitals for t in Timeperiods)

    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

    ## Constraint
    ## Assign the patient to 1 hospital
    def hospital_assignment(model):
        return sum(model.x[j] for j in Hospitals) == 1

    model.hospital_assignment_constr = pyo.Constraint(rule = hospital_assignment)
    
    def available_beds(model, j, t): 
        return model.x[j] <= Beds[j] * model.x[j] - OccupiedBeds[j] * model.x[j] - Arrivals[j, t] * model.x[j] + Discharges[j, t] * model.x[j] + model.penalty[j, t]

    model.available_beds_constr = pyo.Constraint(Hospitals, Timeperiods, rule = available_beds) 

    return model

def main():
    global CensusBlocks, Hospitals, Distance, Beds, OccupiedBeds, opt, solverType, solverVerbose, sched, MEANICUTIME

    parser = argparse.ArgumentParser(prog = appName)
    parser.add_argument('--log', metavar = "level", type = str, default = "info",  choices = ['trace', 'debug', 'info', 'warn', 'error', 'critical'], help = 'set the logging level to the console')
    parser.add_argument('--seed', metavar = "seed", type = int, default = int(time.time()), help = 'set the random seed')
    parser.add_argument('--simtime', metavar = "hours", type = int, default = 90, help = 'number of hours for simulation')
    parser.add_argument('--icutime', metavar = "hours", type = int, default = 10, help = 'mean number of hours for expected ICU time')
    parser.add_argument('--genrate', metavar = "hours", type = float, default = 24, help = 'rate of patient generation')
    parser.add_argument('--hospitaldata', metavar = "file", type = str, default = pkg_resources.resource_filename(__name__, 'data/hospitals.toml'), help = 'use file for hospital data')
    parser.add_argument('--tractdata', metavar = "file", type = str, default = pkg_resources.resource_filename(__name__, 'data/tracts.toml'), help = 'use file for tract data')
    parser.add_argument('--solver', metavar = "name", type = str, default = "glpk",  choices = ['cbc', 'cplex', 'glpk'], help = 'set which solver to use')
    parser.add_argument('--verbose', action = 'store_true', default = False, help = 'display solver output')
    parser.add_argument('--sched', metavar = "scheduler", type = str, default = "home",  choices = ['home', 'nearest', 'lp'], help = 'set the scheduling mechanism')
    parser.add_argument('--patientdata', metavar = "file", type = str, default = None, help = 'write patient data as CSV FILE')
    parser.add_argument('--version', action = 'store_true', default = False, help = 'display version information and quit')

    args = vars(parser.parse_args())
    logLevel = args['log'].lower()
    RANDOM_SEED = args['seed']
    SIM_TIME = args['simtime']
    MEANICUTIME = args['icutime']
    HOSPITALFILE = args['hospitaldata']
    TRACTFILE = args['tractdata']
    solverType = args['solver'].lower()
    solverVerbose = args['verbose']
    sched = args['sched'].lower()
    patientDataFile = args['patientdata']
    patientGenerationRate = 1 / args['genrate'] 
    versionInfo = args['version']

    pyver = sys.version.replace('\n', '')
    versionString = f'Python {pyver}, {pyomo.__name__} {pyomo.__version__}, {simpy.__name__} {simpy.__version__}'
    console.info(f'{infoBanner}, {hospcoop.__name__} v.{hospcoop.__version__}, {appName}')
    console.info(versionString)
    if versionInfo == True:
        sys.exit()

    console.set_level(logLevelMap[logLevel])
    console.info(f'Console logging level {logLevel}')

    console.info(f"Using random seed {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    seed(RANDOM_SEED)

    console.debug(f"Using hospital data from file {HOSPITALFILE}")
    hospitalsData = toml.load(HOSPITALFILE)

    console.debug(f"Using tract data from file {TRACTFILE}")
    tractsData = toml.load(TRACTFILE)

    CensusBlocks = list(tractsData.keys())
    Hospitals = list(hospitalsData.keys())
    for i in CensusBlocks:
        for j in Hospitals:
            Distance[(i, j)] = tractsData[i]["distance"][j]

    for j in Hospitals:
        Beds[j] = hospitalsData[j]['icucount']
        OccupiedBeds[j] = 0

    if sched == "lp":
        console.debug(f"Using solver {solverType}")
        opt = pyo.SolverFactory(solverType)

    # Setup and start the simulation
    console.info("Starting simulation")

    # Create an environment and start the setup process
    env.process(setup(env, patientGenerationRate, hospitalsData, tractsData))

    # Execute!
    env.run(until = SIM_TIME)
    console.info(f"Simulation complete")

    if patientDataFile != None:
        console.info(f"Writing simulated patient data to {patientDataFile}")
        (pd.DataFrame.from_dict(data = patientData, orient = 'index').to_csv(patientDataFile, header = True))
