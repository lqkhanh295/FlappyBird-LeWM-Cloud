import pygame
import numpy as np
from PIL import Image
from .entities import (
    Background,
    Floor,
    Pipes,
    Player,
    PlayerMode,
    Score,
)
from .utils import GameConfig, Images, Sounds, Window

class DummySound:
    def play(self):
        pass

class FlappyEnv:
    def __init__(self, mute=True):
        pygame.init()
        pygame.display.set_caption("Flappy Bird AI")
        self.window = Window(288, 512)
        self.screen = pygame.display.set_mode((self.window.width, self.window.height))
        self.images = Images()
        
        sounds = Sounds()
        if mute:
            # Mute sounds by replacing play methods
            sounds.wing = DummySound()
            sounds.hit = DummySound()
            sounds.die = DummySound()
            sounds.point = DummySound()
            sounds.swoosh = DummySound()

        self.config = GameConfig(
            screen=self.screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=self.window,
            images=self.images,
            sounds=sounds,
        )
        
    def reset(self):
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        
        # Tick a few times to initialize positions
        for _ in range(5):
            self.step(0)
            
        return self.get_obs(), self.get_info()
        
    def step(self, action):
        # Action: 1 is flap, 0 is do nothing
        if action == 1:
            self.player.flap()
            
        # Tick all elements in order (background -> pipes -> floor -> score -> player)
        self.background.tick()
        self.pipes.tick()
        self.floor.tick()
        self.score.tick()
        self.player.tick()
        
        # Check collision
        done = self.player.collided(self.pipes, self.floor)
        
        # Add score when crossing a pipe
        reward = 1.0
        for pipe in self.pipes.upper:
            if self.player.crossed(pipe):
                self.score.add()
                reward += 10.0
                
        pygame.display.update()
        # self.config.tick() # Disable FPS capping to run at maximum speed for data collection / MPC
        
        obs = self.get_obs()
        info = self.get_info()
        
        return obs, reward, done, info
        
    def get_obs(self):
        # Capture screen array
        img = pygame.surfarray.array3d(self.screen)
        # Transpose from (W, H, C) to (H, W, C)
        img = np.transpose(img, (1, 0, 2))
        # Resize to 224x224 (required by ViT encoder)
        pil_img = Image.fromarray(img)
        resized_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
        return np.array(resized_img, dtype=np.uint8)
        
    def get_info(self):
        # Find next pipe
        next_pipe = None
        for pipe in self.pipes.lower:
            if pipe.x + pipe.w > self.player.x:
                next_pipe = pipe
                break
                
        if next_pipe is not None:
            pipe_dist_x = float(next_pipe.x - self.player.x)
            # next_pipe.y is the top of the lower pipe
            pipe_gap_center_y = float(next_pipe.y - self.pipes.pipe_gap / 2)
        else:
            pipe_dist_x = 999.0
            pipe_gap_center_y = float(self.window.viewport_height / 2)
            
        return {
            "proprio": np.array([float(self.player.y), float(self.player.vel_y)], dtype=np.float32),
            "state": np.array([pipe_dist_x, pipe_gap_center_y, float(self.player.y)], dtype=np.float32),
            "score": self.score.score
        }
        
    def close(self):
        pygame.quit()
