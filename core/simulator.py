"""
Physical Layer Simulation Engine

Uses Game Loop pattern to simulate asynchronous transmission
"""

import time
from collections import deque
from typing import Optional
import numpy as np
import logging
import random
import abc


class SimulationEntity(abc.ABC):
    """simulation entity"""
    
    def __init__(self, name: str):
        self.name = name
        self.current_tick = 0
    
    def update(self, tick: int):
        """called every tick"""
        self.current_tick = tick
    
    def reset(self):
        """reset entity state"""
        self.current_tick = 0

    def set_name(self, name:str):
        self.name= name

    def __str__(self):
        return f"SimulationEntity({self.name})"


class PhySimulationEngine:
    """物理层仿真引擎 - 主循环驱动器"""
    
    def __init__(self, time_step_us: float = 1.0, realtime_mode: bool = False):
        """
        初始化仿真引擎
        
        Args:
            time_step_us: 每个tick代表的物理时间（微秒）
            realtime_mode: 是否实时仿真（会按真实时间sleep）
        """
        self.time_step_us = time_step_us
        self.time_step_s = time_step_us * 1e-6
        self.realtime_mode = realtime_mode
        
        self.current_tick = 0
        self.entities: list[SimulationEntity] = []
        self.running = False
        
        # 统计信息
        self.stats = {
            'total_ticks': 0,
            'total_time_s': 0.0,
            'simulation_speed': 0.0  # 仿真速度倍数（仿真时间/墙钟时间）
        }

        self.debug= True

    def set_debug(self, debug: bool):
        self.debug= debug
    
    def register_entity(self, entity: SimulationEntity):
        """注册一个实体到仿真引擎"""
        self.entities.append(entity)
        #print(f"Registered entity: {entity.name}")
    
    def run(self, duration_ticks: int):
        """
        运行仿真
        
        Args:
            duration_ticks: 仿真运行的tick数
        """
        self.running = True
        start_wallclock = time.time()
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"Starting simulation for {duration_ticks} ticks")
            print(f"Time step: {self.time_step_us} μs/tick")
            print(f"Total simulated time: {duration_ticks * self.time_step_us / 1e6:.6f} s")
            print(f"entity num: {len(self.entities)}\n")
            print(f"{'='*60}\n")
        
        for tick in range(duration_ticks):
            self.current_tick = tick
            
            # update all entities
            update_results= [entity.update(tick) for entity in self.entities]
            # 检查是否有实体请求提前结束仿真
            if any(result == 1 for result in update_results):
                print(f"Simulation ending early at tick {tick} due to entity request.")
                break
            
            # 实时模式下控制仿真速度
            if self.realtime_mode:
                time.sleep(self.time_step_s)
            
            # 定期打印进度
            # if (tick + 1) % 1000 == 0:
            #     progress = (tick + 1) / duration_ticks * 100
            #     print(f"Progress: {progress:.1f}% (tick {tick+1}/{duration_ticks})")
        
        # 计算统计信息
        end_wallclock = time.time()
        wallclock_time = end_wallclock - start_wallclock
        simulated_time = duration_ticks * self.time_step_s
        
        self.stats['total_ticks'] = duration_ticks
        self.stats['total_time_s'] = simulated_time
        self.stats['simulation_speed'] = simulated_time / wallclock_time if wallclock_time > 0 else 0
        

        if self.debug:
            print(f"\n{'='*60}")
            print(f"Simulation completed!")
            print(f"Simulated time: {simulated_time:.6f} s")
            print(f"Wallclock time: {wallclock_time:.6f} s")
            print(f"Simulation speed: {self.stats['simulation_speed']:.2f}x realtime")
            print(f"{'='*60}\n")

        # self.reset()
        
        self.running = False
    
    def reset(self):
        """重置仿真"""
        self.current_tick = 0
        for entity in self.entities:
            entity.reset()
        print("Simulation reset")


# ============================================================================
# 简单测试
# ============================================================================

def test_engine():
    """测试仿真引擎基本功能"""
    
    # 创建一个简单的测试实体
    class DummyEntity(SimulationEntity):
        def __init__(self, name: str, enermy: Optional['DummyEntity'] = None):
            super().__init__(name)
            self.update_count = 0
            self.health= 100
            self.enermy = enermy

        def set_enermy(self, enermy: 'DummyEntity'):
            self.enermy = enermy

        def attack(self, damage: int):
            if self.enermy:
                self.enermy.health -= damage if self.enermy.health >= damage else 0
                print(f"{self.name} attacks {self.enermy.name} for {damage} damage! {self.enermy.name} health: {self.enermy.health}")

        def update(self, tick: int):
            super().update(tick)
            self.update_count += 1
            if self.health <= 0:
                print(f"{self.name} has been defeated!")
                return 1
            if self.update_count % 5000 == 0:
                #print(f"{self.name}: update called {self.update_count} times")
                if self.enermy:
                    damage = random.randint(5, 15)
                    self.attack(damage)
    
    # 创建仿真引擎
    engine = PhySimulationEngine(time_step_us=1.0, realtime_mode=False)
    
    # 注册测试实体
    entity1 = DummyEntity("Entity-A")
    entity2 = DummyEntity("Entity-B")
    entity1.set_enermy(entity2)
    entity2.set_enermy(entity1)
    engine.register_entity(entity1)
    engine.register_entity(entity2)
    
    # 运行仿真
    engine.run(duration_ticks=100000)
    


if __name__ == "__main__":
    test_engine()