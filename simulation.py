import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# -------------------------

import pygame
import random
import time
from ultralytics import YOLO


MODEL_PATH = 'runs/detect/NEU_defect_model_v1/weights/best.pt'
TEST_IMAGES_DIR = 'test_images'

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 400
BELT_SPEED = 2
INSPECTION_INTERVAL = 4


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BLUE_GRAY = (100, 120, 140)
GREEN = (0, 200, 0)
YELLOW = (255, 230, 0)
RED = (220, 0, 0)


class QualityControlAgent:
    """
    An agent that uses a trained YOLOv8 model to inspect parts
    and make a decision based on the detected defects.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Please check the path.")
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print("Model loaded successfully.")
        print(f"Model classes: {self.class_names}")

    def inspect_part(self, image_path):
        results = self.model(image_path, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0:
            return {"decision": "Pass", "summary": "No defects detected."}
        else:
            return self._analyze_defects(result)

    def _analyze_defects(self, result):
        """Analyzes the detected defects and makes a final decision."""
        summary_lines = []
        is_critical = False
        is_major = False
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            defect_name = self.class_names[class_id]
            summary_lines.append(f"- Found '{defect_name}' with {confidence:.2f} confidence.")
            
            if defect_name in ['patches', 'rolled-in_scale']:
                is_critical = True
            if defect_name in ['scratches', 'crazing'] and confidence > 0.7:
                is_major = True

        # --- Final Decision Hierarchy (CORRECTED SECTION) ---
        summary = "\n".join(summary_lines)
        if is_critical:
            return {"decision": "Halt & Alert", "summary": summary}
        elif is_major:
            return {"decision": "Reroute", "summary": summary}
        else:
            return {"decision": "Log & Pass", "summary": summary}

def run_simulation():
    """Main function to run the PyGame simulation."""
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Agentic Quality Control Simulation")
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    try:
        agent = QualityControlAgent(model_path=MODEL_PATH)
        test_images = [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
        if not test_images:
            raise FileNotFoundError(f"No images found in the '{TEST_IMAGES_DIR}' folder.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    part_x = -100
    part_color = BLUE_GRAY
    last_inspection_time = time.time()
    current_decision = "Waiting..."
    current_summary = "No part under inspection."
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        part_x += BELT_SPEED
        if part_x > SCREEN_WIDTH:
            part_x = -100

        inspection_station_x = SCREEN_WIDTH // 2
        current_time = time.time()
        
        if inspection_station_x - 50 < part_x < inspection_station_x + 50 and (current_time - last_inspection_time) > INSPECTION_INTERVAL:
            print("\nPart at inspection station. Running analysis...")
            test_image = random.choice(test_images)
            print(f"Inspecting image: {test_image}")

            result = agent.inspect_part(test_image)
            current_decision = result['decision']
            current_summary = result['summary']
            
            print(f"Decision: {current_decision}")
            
            if current_decision == "Pass":
                part_color = GREEN
            elif current_decision in ["Log & Pass", "Reroute"]:
                part_color = YELLOW
            elif current_decision == "Halt & Alert":
                part_color = RED
            
            last_inspection_time = current_time

        screen.fill(GRAY)
        pygame.draw.rect(screen, BLACK, (0, SCREEN_HEIGHT // 2 - 50, SCREEN_WIDTH, 100))
        pygame.draw.rect(screen, (255, 255, 100, 50), (inspection_station_x - 75, 0, 150, SCREEN_HEIGHT), 2)
        station_text = small_font.render("INSPECTION", True, WHITE)
        screen.blit(station_text, (inspection_station_x - 50, 10))
        
        part_rect = pygame.Rect(part_x, SCREEN_HEIGHT // 2 - 25, 100, 50)
        pygame.draw.rect(screen, part_color, part_rect)
        pygame.draw.rect(screen, WHITE, part_rect, 2)

        decision_text = font.render(f"Agent Decision: {current_decision}", True, WHITE)
        screen.blit(decision_text, (20, SCREEN_HEIGHT - 80))
        summary_text = small_font.render(f"Summary: {current_summary.splitlines()[0]}", True, WHITE)
        screen.blit(summary_text, (20, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    run_simulation()