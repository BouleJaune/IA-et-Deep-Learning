import sc2
from sc2 import run_game, maps, Race, Difficulty
import random
from sc2.player import Bot, Computer
from sc2.constants import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

class MonBotProtoss(sc2.BotAI):
    
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.max_probes = 70
        self.ScoreUnits = {"t": [], "y": []}
        self.ScoreMineRate = {"t": [], "y": []}
        self.ScoreVespeneRate = {"t": [], "y": []}
        
    async def on_step(self, iteration):
        self.temps_minute = iteration / self.ITERATIONS_PER_MINUTE
        await self.distribute_workers()
        await self.build_workers()
        await self.build_supply()
        await self.build_gas()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        # await self.defend()
        await self.attack()
        await self.minimap()
        await self.recupscore()
        
    async def minimap(self):
        game_data = np.zeros((self.game_info.map_size[1]*8, self.game_info.map_size[0]*8, 3), np.uint8)
                   
        for Unit in self.units():
            if(Unit.is_ready):
                pos = Unit.position
                cv2.circle(game_data, (int(pos[0]*8), int(pos[1]*8)),  int(Unit.radius*8), (255,0,0), -1)
      
        for Unit in self.known_enemy_units():
            if(Unit.is_ready):
                pos = Unit.position
                cv2.circle(game_data, (int(pos[0]*8), int(pos[1]*8)),  int(Unit.radius*8), (0,0,255), -1)
        
            
        flipped = cv2.flip(game_data, 0)       
        cv2.imshow("MiniMap", flipped)
        cv2.waitKey(1)
        
    async def recupscore(self):        
    
        s = self.state.score.total_value_units #renvoie un int croissant, soit récup le final soit tracer la courbe d'évolution
        t = self.temps_minute*60
        self.ScoreUnits["t"].append(t)
        self.ScoreUnits["y"].append(s)        
        
        s = self.state.score.collection_rate_minerals
        t = self.temps_minute*60
        self.ScoreMineRate["t"].append(t)
        self.ScoreMineRate["y"].append(s)
        
        s = self.state.score.collection_rate_vespene
        t = self.temps_minute*60
        self.ScoreVespeneRate["t"].append(t)
        self.ScoreVespeneRate["y"].append(s)
        
       
    # train data => des choix à faire, labels => les choix habiles car victoire, features => la minimap, on créer la donnée que si victoire "que si victoire" pas ouf
    # utilisation => au moment de faire un choix, on fait une prédiction avec comme entrée l'état actuel de la minimap
    # idée => avoir de la mémoire
    # idée => faire plusieurs modèles pour plusieurs types de choix
    # idée => des variables hardcodables mais à opti avec du genre "best perf sur x games"
    # idée => pour opti la macro créer les choix possibles et entrainer sur les scores de macro
    # idée => pour opti la micro créer les choix possibles et entrainer sur les scores de micro (type unités perdues, vie restante etc)
    # idée => rajouter du  deep reinforcement learning
       
        
        
        
        
    async def build_workers(self):
        if self.units(NEXUS).amount * 22 > self.units(PROBE).amount:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE) and self.units(PROBE).amount < self.max_probes:
                    await self.do(nexus.train(PROBE))     
        
    async def build_supply(self):
        if self.supply_left < 5 and not self.already_pending(PYLON) and self.supply_cap < 200:       
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)
                    if self.can_afford(PYLON) and self.supply_cap > 60:
                        await self.build(PYLON, near=nexuses.first)
    
    async def build_gas(self):            
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(10.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                if self.supply_left < 3:
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))
                    
    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS) and self.temps_minute > 8 :
            await self.expand_now()
        
    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists:
                if not self.units(CYBERNETICSCORE):
                    if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                        await self.build(CYBERNETICSCORE, near=pylon)
                        
                if self.units(GATEWAY).amount < 4*self.units(NEXUS).amount:
                    await self.build(GATEWAY, near=pylon)
                    
            elif self.can_afford(GATEWAY) and self.units(GATEWAY).amount < 2:                
                    await self.build(GATEWAY, near=pylon)
                
    async def build_council(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if not self.units(TWILIGHTCOUNCIL).ready.exists:
                if self.can_afford(TWILIGHTCOUNCIL) and not self.already_pending(TWILIGHTCOUNCIL):
                    await self.build(TWILIGHTCOUNCIL, near=pylon)

                
    async def research_blink(self):
        if self.can_afford(RESEARCH_BLINK):
            council = self.units(TWILIGHTCOUNCIL).ready.random
            await self.do(council.research(BLINKTECH))


    async def build_offensive_force(self):
        if self.units(CYBERNETICSCORE).ready.exists:
            for gw in self.units(GATEWAY).noqueue:        
                if self.can_afford(STALKER) and self.supply_left > 2 and gw.is_ready:                
                    if self.units(STALKER).amount < 12:
                        await self.do(gw.train(STALKER))        
                    elif not self.units(TWILIGHTCOUNCIL).ready.exists:
                        await self.build_council()
                    elif self.already_pending_upgrade(BLINKTECH)==0:                                
                        await self.research_blink()
                    else:
                        await self.do(gw.train(STALKER))
            
                

          

                
    async def defend(self):
        if self.units(STALKER).amount > 3:
            if len(self.known_enemy_units) > 0:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))
    
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return random.choice(self.enemy_start_locations)
    
    async def attack(self):
        if self.units(STALKER).amount > 12:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(self.find_target(self.state)))
    
    
    
    
lebot=MonBotProtoss()
run_game(maps.get("AbyssalReefLE"), 
    [Bot(Race.Protoss, lebot), Computer(Race.Terran, Difficulty.VeryHard)], realtime=False)

plt.plot(lebot.ScoreUnits["t"], lebot.ScoreUnits["y"], label="Score Unités",color="red"   )
plt.plot(lebot.ScoreVespeneRate["t"], lebot.ScoreVespeneRate["y"], label="Score Vespene rate",color="green")
plt.plot(lebot.ScoreMineRate["t"], lebot.ScoreMineRate["y"], label="Score Minerals rate",color="blue")
plt.legend()
plt.show()
  #difficulties : VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, CheatMoney, CheatInsane