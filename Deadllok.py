import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import random
import math
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import time

# Initialize GLUT for text rendering
glutInit()

class NodeType(Enum):
    PROCESS = 0
    RESOURCE = 1

class EdgeType(Enum):
    REQUEST = 0  # Process -> Resource (dashed, red)
    ALLOCATION = 1  # Resource -> Process (solid, green)

@dataclass
class Node:
    id: int
    type: NodeType
    x: float
    y: float
    target_x: float
    target_y: float
    label: str
    instances: int = 1  # For resources: total instances
    allocated: int = 0  # For resources: currently allocated
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    selected: bool = False
    pulse: float = 0.0  # For animation effects
    
    def update_position(self, dt: float):
        # Smooth interpolation to target position
        speed = 5.0 * dt
        self.x += (self.target_x - self.x) * speed
        self.y += (self.target_y - self.y) * speed
        self.pulse += dt * 3.0

@dataclass
class Edge:
    source: int  # Node ID
    target: int  # Node ID
    type: EdgeType
    instances: int = 1  # Number of resource instances
    animated_offset: float = 0.0
    
    def update(self, dt: float):
        self.animated_offset += dt * 2.0

class DeadlockDetector:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.next_id = 0
        self.deadlock_cycle: List[int] = []
        self.is_deadlocked = False
        self.simulation_time = 0.0
        
    def add_process(self, x: float, y: float) -> int:
        node_id = self.next_id
        self.next_id += 1
        color = (0.3, 0.6, 1.0)  # Blue for processes
        self.nodes[node_id] = Node(
            id=node_id,
            type=NodeType.PROCESS,
            x=x, y=y,
            target_x=x, target_y=y,
            label=f"P{node_id}",
            color=color
        )
        return node_id
    
    def add_resource(self, x: float, y: float, instances: int = 1) -> int:
        node_id = self.next_id
        self.next_id += 1
        color = (1.0, 0.5, 0.2)  # Orange for resources
        self.nodes[node_id] = Node(
            id=node_id,
            type=NodeType.RESOURCE,
            x=x, y=y,
            target_x=x, target_y=y,
            label=f"R{node_id}",
            instances=instances,
            allocated=0,
            color=color
        )
        return node_id
    
    def add_request_edge(self, process_id: int, resource_id: int):
        # Check if edge already exists
        for edge in self.edges:
            if edge.source == process_id and edge.target == resource_id and edge.type == EdgeType.REQUEST:
                return
        self.edges.append(Edge(process_id, resource_id, EdgeType.REQUEST))
    
    def add_allocation_edge(self, resource_id: int, process_id: int, instances: int = 1):
        # Check if edge already exists
        for edge in self.edges:
            if edge.source == resource_id and edge.target == process_id and edge.type == EdgeType.ALLOCATION:
                return
        self.edges.append(Edge(resource_id, process_id, EdgeType.ALLOCATION, instances))
        # Update allocated count
        if resource_id in self.nodes:
            self.nodes[resource_id].allocated += instances
    
    def remove_edge(self, source: int, target: int, edge_type: EdgeType):
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target and e.type == edge_type)]
        if edge_type == EdgeType.ALLOCATION:
            if source in self.nodes:
                self.nodes[source].allocated = max(0, self.nodes[source].allocated - 1)
    
    def detect_deadlock(self) -> bool:
        # Build wait-for graph
        # Process waits for Process if Process holds Resource that first Process requests
        wait_for: Dict[int, Set[int]] = {node_id: set() for node_id in self.nodes 
                                        if self.nodes[node_id].type == NodeType.PROCESS}
        
        # Find all request edges (Process -> Resource)
        requests = [(e.source, e.target) for e in self.edges if e.type == EdgeType.REQUEST]
        # Find all allocation edges (Resource -> Process)
        allocations = [(e.source, e.target) for e in self.edges if e.type == EdgeType.ALLOCATION]
        
        # Build mapping: Resource -> [Processes holding it]
        resource_holders: Dict[int, List[int]] = {}
        for res_id, proc_id in allocations:
            if res_id not in resource_holders:
                resource_holders[res_id] = []
            resource_holders[res_id].append(proc_id)
        
        # Build wait-for graph: Process A waits for Process B if A requests resource held by B
        for proc_id, res_id in requests:
            if res_id in resource_holders:
                for holder_id in resource_holders[res_id]:
                    if holder_id != proc_id:
                        wait_for[proc_id].add(holder_id)
        
        # Cycle detection using DFS
        visited = set()
        rec_stack = set()
        cycle_path = []
        
        def dfs(node: int, path: List[int]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in wait_for.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    self.deadlock_cycle = path[cycle_start:] + [neighbor]
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for process_id in wait_for:
            if process_id not in visited:
                if dfs(process_id, []):
                    self.is_deadlocked = True
                    return True
        
        self.is_deadlocked = False
        self.deadlock_cycle = []
        return False
    
    def update_layout(self, dt: float):
        # Force-directed layout
        width, height = 800, 600
        center_x, center_y = width / 2, height / 2
        
        # Separate processes and resources
        processes = [n for n in self.nodes.values() if n.type == NodeType.PROCESS]
        resources = [n for n in self.nodes.values() if n.type == NodeType.RESOURCE]
        
        # Position processes on left, resources on right in a circular arrangement
        if processes:
            angle_step = 2 * math.pi / max(len(processes), 1)
            radius = 200
            for i, proc in enumerate(processes):
                angle = math.pi + i * angle_step  # Left side
                proc.target_x = center_x + radius * math.cos(angle) - 150
                proc.target_y = center_y + radius * math.sin(angle) * 0.6
        
        if resources:
            angle_step = 2 * math.pi / max(len(resources), 1)
            radius = 200
            for i, res in enumerate(resources):
                angle = i * angle_step  # Right side
                res.target_x = center_x + radius * math.cos(angle) + 150
                res.target_y = center_y + radius * math.sin(angle) * 0.6
        
        # Update positions
        for node in self.nodes.values():
            node.update_position(dt)
        
        # Update edges
        for edge in self.edges:
            edge.update(dt)
        
        self.simulation_time += dt
    
    def get_node_at(self, x: float, y: float, radius: float = 30.0) -> Optional[int]:
        for node_id, node in self.nodes.items():
            dx = node.x - x
            dy = node.y - y
            if math.sqrt(dx*dx + dy*dy) < radius:
                return node_id
        return None

class OpenGLRenderer:
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.detector = DeadlockDetector()
        self.selected_node: Optional[int] = None
        self.dragging_node: Optional[int] = None
        self.mode = "normal"  # normal, adding_request, adding_allocation
        self.source_node: Optional[int] = None
        self.show_help = True
        self.auto_simulate = False
        self.simulation_timer = 0.0
        
        # Initialize Pygame and OpenGL
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Deadlock Detection Visualization System - OpenGL")
        
        # OpenGL setup
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Font setup
        self.font = pygame.font.SysFont('monospace', 14)
        self.title_font = pygame.font.SysFont('monospace', 24, bold=True)
        
        # Create initial example
        self.create_example_scenario()
    
    def create_example_scenario(self):
        # Create a classic deadlock scenario
        # P0 holds R0, requests R1
        # P1 holds R1, requests R0
        
        p0 = self.detector.add_process(300, 300)
        p1 = self.detector.add_process(300, 500)
        r0 = self.detector.add_resource(700, 300, instances=1)
        r1 = self.detector.add_resource(700, 500, instances=1)
        
        # Create deadlock
        self.detector.add_allocation_edge(r0, p0)
        self.detector.add_request_edge(p0, r1)
        self.detector.add_allocation_edge(r1, p1)
        self.detector.add_request_edge(p1, r0)
        
        self.detector.detect_deadlock()
    
    def draw_circle(self, x: float, y: float, radius: float, color: Tuple[float, float, float], 
                    filled: bool = True, segments: int = 32):
        if filled:
            glBegin(GL_TRIANGLE_FAN)
        else:
            glBegin(GL_LINE_LOOP)
        glColor3f(*color)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex2f(x + radius * math.cos(angle), y + radius * math.sin(angle))
        glEnd()
    
    def draw_rectangle(self, x: float, y: float, width: float, height: float, 
                       color: Tuple[float, float, float], filled: bool = True):
        if filled:
            glBegin(GL_QUADS)
        else:
            glBegin(GL_LINE_LOOP)
        glColor3f(*color)
        glVertex2f(x - width/2, y - height/2)
        glVertex2f(x + width/2, y - height/2)
        glVertex2f(x + width/2, y + height/2)
        glVertex2f(x - width/2, y + height/2)
        glEnd()
    
    def draw_dashed_line(self, x1: float, y1: float, x2: float, y2: float, 
                         color: Tuple[float, float, float], dash_length: float = 10.0,
                         animated_offset: float = 0.0):
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0:
            return
        
        dx /= dist
        dy /= dist
        
        glColor3f(*color)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        
        offset = animated_offset % (dash_length * 2)
        current_dist = offset
        drawing = True
        
        while current_dist < dist:
            start = current_dist
            end = min(current_dist + dash_length, dist)
            
            if drawing:
                glVertex2f(x1 + dx * start, y1 + dy * start)
                glVertex2f(x1 + dx * end, y1 + dy * end)
            
            current_dist += dash_length
            drawing = not drawing
        
        glEnd()
    
    def draw_arrow(self, x1: float, y1: float, x2: float, y2: float, 
                   color: Tuple[float, float, float], head_size: float = 15.0,
                   dashed: bool = False, animated_offset: float = 0.0):
        if dashed:
            self.draw_dashed_line(x1, y1, x2, y2, color, animated_offset=animated_offset)
        else:
            glColor3f(*color)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            glVertex2f(x1, y1)
            glVertex2f(x2, y2)
            glEnd()
        
        # Draw arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        # Offset arrowhead to not overlap with node
        node_radius = 35.0
        arrow_x = x2 - node_radius * math.cos(angle)
        arrow_y = y2 - node_radius * math.sin(angle)
        
        glColor3f(*color)
        glBegin(GL_TRIANGLES)
        glVertex2f(arrow_x, arrow_y)
        glVertex2f(arrow_x - head_size * math.cos(angle - math.pi/6), 
                   arrow_y - head_size * math.sin(angle - math.pi/6))
        glVertex2f(arrow_x - head_size * math.cos(angle + math.pi/6), 
                   arrow_y - head_size * math.sin(angle + math.pi/6))
        glEnd()
    
    def draw_text(self, text: str, x: float, y: float, color: Tuple[int, int, int] = (255, 255, 255),
                  center: bool = True, font=None):
        if font is None:
            font = self.font
        
        surface = font.render(text, True, color)
        text_data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()
        
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        if center:
            x -= width / 2
            y -= height / 2
        glTexCoord2f(0, 0)
        glVertex2f(x, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + width, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1)
        glVertex2f(x, y + height)
        glEnd()
        glDisable(GL_TEXTURE_2D)
    
    def draw_glow(self, x: float, y: float, radius: float, color: Tuple[float, float, float], intensity: float = 1.0):
        # Draw pulsing glow effect
        for i in range(5, 0, -1):
            alpha = 0.1 * intensity * (6 - i) / 5
            r = radius + i * 8
            glColor4f(color[0], color[1], color[2], alpha)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(x, y)
            for j in range(32):
                angle = 2 * math.pi * j / 32
                glVertex2f(x + r * math.cos(angle), y + r * math.sin(angle))
            glEnd()
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # Background gradient effect
        glBegin(GL_QUADS)
        glColor3f(0.05, 0.05, 0.1)
        glVertex2f(0, 0)
        glVertex2f(self.width, 0)
        glColor3f(0.1, 0.1, 0.15)
        glVertex2f(self.width, self.height)
        glVertex2f(0, self.height)
        glEnd()
        
        # Draw title
        self.draw_text("DEADLOCK DETECTION VISUALIZATION SYSTEM", self.width/2, 40, 
                      (100, 200, 255), font=self.title_font)
        
        # Draw edges first (behind nodes)
        for edge in self.detector.edges:
            source = self.detector.nodes[edge.source]
            target = self.detector.nodes[edge.target]
            
            if edge.type == EdgeType.REQUEST:
                color = (1.0, 0.3, 0.3)  # Red for request
                dashed = True
            else:
                color = (0.3, 1.0, 0.3)  # Green for allocation
                dashed = False
            
            # Highlight if part of deadlock cycle
            if self.detector.is_deadlocked:
                cycle_ids = self.detector.deadlock_cycle
                if source.id in cycle_ids and target.id in cycle_ids:
                    color = (1.0, 0.0, 0.0)  # Bright red for deadlock
                    self.draw_glow((source.x + target.x)/2, (source.y + target.y)/2, 20, (1.0, 0.0, 0.0))
            
            self.draw_arrow(source.x, source.y, target.x, target.y, color, 
                          dashed=dashed, animated_offset=edge.animated_offset)
        
        # Draw temporary edge when adding connection
        if self.mode in ["adding_request", "adding_allocation"] and self.source_node is not None:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            source = self.detector.nodes[self.source_node]
            if self.mode == "adding_request":
                color = (1.0, 0.5, 0.5)
            else:
                color = (0.5, 1.0, 0.5)
            self.draw_arrow(source.x, source.y, mouse_x, mouse_y, color, dashed=True)
        
        # Draw nodes
        for node_id, node in self.detector.nodes.items():
            # Glow effect for selected or deadlocked nodes
            if node.selected:
                pulse = (math.sin(node.pulse) + 1) / 2
                self.draw_glow(node.x, node.y, 40, (1.0, 1.0, 0.0), intensity=0.5 + pulse * 0.5)
            elif self.detector.is_deadlocked and node.id in self.detector.deadlock_cycle:
                pulse = (math.sin(node.pulse * 3) + 1) / 2
                self.draw_glow(node.x, node.y, 45, (1.0, 0.0, 0.0), intensity=0.7 + pulse * 0.3)
            
            # Draw node shape
            if node.type == NodeType.PROCESS:
                # Circle for process
                base_color = node.color
                if self.detector.is_deadlocked and node.id in self.detector.deadlock_cycle:
                    base_color = (1.0, 0.2, 0.2)
                self.draw_circle(node.x, node.y, 30, base_color)
                self.draw_circle(node.x, node.y, 30, (0.8, 0.8, 0.8), filled=False)
            else:
                # Square for resource
                base_color = node.color
                if self.detector.is_deadlocked and node.id in self.detector.deadlock_cycle:
                    base_color = (1.0, 0.2, 0.2)
                self.draw_rectangle(node.x, node.y, 50, 50, base_color)
                self.draw_rectangle(node.x, node.y, 50, 50, (0.8, 0.8, 0.8), filled=False)
                
                # Draw instance dots
                total = node.instances
                allocated = node.allocated
                available = total - allocated
                
                dot_y = node.y - 15
                for i in range(total):
                    if i < available:
                        dot_color = (0.0, 1.0, 0.0)  # Green for available
                    else:
                        dot_color = (1.0, 0.0, 0.0)  # Red for allocated
                    dot_x = node.x - (total-1)*6 + i*12
                    self.draw_circle(dot_x, dot_y, 4, dot_color)
            
            # Draw label
            text_color = (255, 255, 255) if not (self.detector.is_deadlocked and node.id in self.detector.deadlock_cycle) else (255, 200, 200)
            self.draw_text(node.label, node.x, node.y, text_color)
        
        # Draw UI Panel
        self.draw_ui_panel()
        
        pygame.display.flip()
    
    def draw_ui_panel(self):
        panel_x = 20
        panel_y = 80
        panel_width = 250
        panel_height = 500
        
        # Semi-transparent background
        glColor4f(0.1, 0.1, 0.2, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        glColor3f(0.5, 0.5, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Status
        y_offset = panel_y + 30
        if self.detector.is_deadlocked:
            self.draw_text("⚠️ DEADLOCK DETECTED!", panel_x + panel_width/2, y_offset, (255, 100, 100))
        else:
            self.draw_text("✓ System Safe", panel_x + panel_width/2, y_offset, (100, 255, 100))
        
        y_offset += 40
        self.draw_text(f"Processes: {len([n for n in self.detector.nodes.values() if n.type == NodeType.PROCESS])}", 
                      panel_x + 10, y_offset, (200, 200, 200), center=False)
        y_offset += 25
        self.draw_text(f"Resources: {len([n for n in self.detector.nodes.values() if n.type == NodeType.RESOURCE])}", 
                      panel_x + 10, y_offset, (200, 200, 200), center=False)
        y_offset += 25
        self.draw_text(f"Edges: {len(self.detector.edges)}", 
                      panel_x + 10, y_offset, (200, 200, 200), center=False)
        
        # Controls
        y_offset += 50
        self.draw_text("CONTROLS:", panel_x + panel_width/2, y_offset, (255, 255, 150))
        y_offset += 30
        
        controls = [
            "P - Add Process",
            "R - Add Resource", 
            "Click - Select/Move",
            "Q - Request Edge (P→R)",
            "A - Allocate Edge (R→P)",
            "DEL - Remove Selected",
            "C - Clear All",
            "SPACE - Auto-Simulate",
            "H - Toggle Help",
            "ESC - Exit"
        ]
        
        for control in controls:
            y_offset += 22
            self.draw_text(control, panel_x + 10, y_offset, (180, 180, 180), center=False)
        
        # Current mode
        y_offset += 40
        mode_text = f"Mode: {self.mode.upper()}"
        if self.mode == "adding_request":
            mode_text += " (Select Resource)"
        elif self.mode == "adding_allocation":
            mode_text += " (Select Process)"
        self.draw_text(mode_text, panel_x + panel_width/2, y_offset, (255, 200, 100))
        
        # Help overlay
        if self.show_help:
            self.draw_help_overlay()
    
    def draw_help_overlay(self):
        overlay_x = 300
        overlay_y = 150
        overlay_width = 600
        overlay_height = 400
        
        # Dark background
        glColor4f(0.0, 0.0, 0.0, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(overlay_x, overlay_y)
        glVertex2f(overlay_x + overlay_width, overlay_y)
        glVertex2f(overlay_x + overlay_width, overlay_y + overlay_height)
        glVertex2f(overlay_x, overlay_y + overlay_height)
        glEnd()
        
        glColor3f(0.8, 0.8, 1.0)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(overlay_x, overlay_y)
        glVertex2f(overlay_x + overlay_width, overlay_y)
        glVertex2f(overlay_x + overlay_width, overlay_y + overlay_height)
        glVertex2f(overlay_x, overlay_y + overlay_height)
        glEnd()
        
        # Help content
        y = overlay_y + 40
        x = overlay_x + 30
        
        self.draw_text("DEADLOCK DETECTION SYSTEM - HELP", overlay_x + overlay_width/2, y, (100, 200, 255), 
                      font=self.title_font)
        
        y += 60
        help_texts = [
            "GRAPH ELEMENTS:",
            "  • Blue Circles = Processes (P0, P1, ...)",
            "  • Orange Squares = Resources (R0, R1, ...)",
            "  • Red Dashed Arrows = Request edges (Process wants Resource)",
            "  • Green Solid Arrows = Allocation edges (Resource held by Process)",
            "",
            "HOW TO CREATE DEADLOCK:",
            "  1. Create 2+ Processes (Press P, click to place)",
            "  2. Create 2+ Resources (Press R, click to place)",
            "  3. Add allocation: Select Resource, press A, click Process",
            "  4. Add request: Select Process, press Q, click Resource",
            "  5. Create cycle: P0→R1→P1→R0→P0",
            "",
            "The system automatically detects cycles and highlights deadlocked nodes in RED!"
        ]
        
        for line in help_texts:
            self.draw_text(line, x, y, (220, 220, 220), center=False)
            y += 25
        
        self.draw_text("Press H to close this help", overlay_x + overlay_width/2, overlay_y + overlay_height - 30, 
                      (150, 150, 255))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_p:
                    # Add process at random position
                    x = random.randint(400, 800)
                    y = random.randint(200, 600)
                    self.detector.add_process(x, y)
                elif event.key == pygame.K_r:
                    # Add resource at random position
                    x = random.randint(400, 800)
                    y = random.randint(200, 600)
                    self.detector.add_resource(x, y, instances=random.randint(1, 3))
                elif event.key == pygame.K_c:
                    # Clear all
                    self.detector = DeadlockDetector()
                    self.selected_node = None
                elif event.key == pygame.K_q:
                    # Start adding request edge (Process -> Resource)
                    if self.selected_node is not None:
                        node = self.detector.nodes[self.selected_node]
                        if node.type == NodeType.PROCESS:
                            self.mode = "adding_request"
                            self.source_node = self.selected_node
                        else:
                            print("Select a Process first to create request edge")
                    else:
                        print("Select a Process first")
                elif event.key == pygame.K_a:
                    # Start adding allocation edge (Resource -> Process)
                    if self.selected_node is not None:
                        node = self.detector.nodes[self.selected_node]
                        if node.type == NodeType.RESOURCE:
                            self.mode = "adding_allocation"
                            self.source_node = self.selected_node
                        else:
                            print("Select a Resource first to create allocation edge")
                    else:
                        print("Select a Resource first")
                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                    # Remove selected node
                    if self.selected_node is not None:
                        # Remove edges connected to this node
                        self.detector.edges = [e for e in self.detector.edges 
                                             if e.source != self.selected_node and e.target != self.selected_node]
                        # Remove node
                        del self.detector.nodes[self.selected_node]
                        self.selected_node = None
                        self.detector.detect_deadlock()
                elif event.key == pygame.K_SPACE:
                    self.auto_simulate = not self.auto_simulate
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = event.pos
                    
                    # Check if clicking on a node
                    clicked_node = self.detector.get_node_at(mouse_x, mouse_y)
                    
                    if self.mode == "adding_request" and self.source_node is not None:
                        if clicked_node is not None:
                            target = self.detector.nodes[clicked_node]
                            if target.type == NodeType.RESOURCE:
                                self.detector.add_request_edge(self.source_node, clicked_node)
                                self.detector.detect_deadlock()
                        self.mode = "normal"
                        self.source_node = None
                    
                    elif self.mode == "adding_allocation" and self.source_node is not None:
                        if clicked_node is not None:
                            target = self.detector.nodes[clicked_node]
                            if target.type == NodeType.PROCESS:
                                res = self.detector.nodes[self.source_node]
                                if res.allocated < res.instances:
                                    self.detector.add_allocation_edge(self.source_node, clicked_node)
                                    self.detector.detect_deadlock()
                                else:
                                    print("No available instances of this resource!")
                        self.mode = "normal"
                        self.source_node = None
                    
                    else:
                        if clicked_node is not None:
                            # Select node
                            if self.selected_node is not None:
                                self.detector.nodes[self.selected_node].selected = False
                            self.selected_node = clicked_node
                            self.detector.nodes[clicked_node].selected = True
                            self.dragging_node = clicked_node
                        else:
                            # Deselect
                            if self.selected_node is not None:
                                self.detector.nodes[self.selected_node].selected = False
                                self.selected_node = None
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_node = None
            
            if event.type == pygame.MOUSEMOTION:
                if self.dragging_node is not None:
                    self.detector.nodes[self.dragging_node].target_x = event.pos[0]
                    self.detector.nodes[self.dragging_node].target_y = event.pos[1]
                    self.detector.nodes[self.dragging_node].x = event.pos[0]
                    self.detector.nodes[self.dragging_node].y = event.pos[1]
        
        return True
    
    def auto_simulation_step(self):
        """Randomly modify the graph to show dynamic deadlock detection"""
        processes = [n for n in self.detector.nodes.values() if n.type == NodeType.PROCESS]
        resources = [n for n in self.detector.nodes.values() if n.type == NodeType.RESOURCE]
        
        if not processes or not resources:
            return
        
        action = random.choice(["add_request", "add_allocation", "remove_edge", "add_process", "add_resource"])
        
        if action == "add_request" and processes and resources:
            p = random.choice(processes)
            r = random.choice(resources)
            # Check if not already requesting
            existing = [e for e in self.detector.edges if e.source == p.id and e.target == r.id and e.type == EdgeType.REQUEST]
            if not existing:
                self.detector.add_request_edge(p.id, r.id)
                
        elif action == "add_allocation" and processes and resources:
            r = random.choice(resources)
            p = random.choice(processes)
            if r.allocated < r.instances:
                existing = [e for e in self.detector.edges if e.source == r.id and e.target == p.id and e.type == EdgeType.ALLOCATION]
                if not existing:
                    self.detector.add_allocation_edge(r.id, p.id)
                    
        elif action == "remove_edge" and self.detector.edges:
            edge = random.choice(self.detector.edges)
            self.detector.remove_edge(edge.source, edge.target, edge.type)
            
        elif action == "add_process":
            x = random.randint(200, 1000)
            y = random.randint(150, 700)
            self.detector.add_process(x, y)
            
        elif action == "add_resource":
            x = random.randint(200, 1000)
            y = random.randint(150, 700)
            self.detector.add_resource(x, y, instances=random.randint(1, 2))
        
        self.detector.detect_deadlock()
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            dt = clock.tick(60) / 1000.0  # Delta time in seconds
            
            running = self.handle_events()
            
            # Auto-simulation
            if self.auto_simulate:
                self.simulation_timer += dt
                if self.simulation_timer > 2.0:  # Every 2 seconds
                    self.auto_simulation_step()
                    self.simulation_timer = 0.0
            
            # Update layout and animations
            self.detector.update_layout(dt)
            
            # Render
            self.render()
        
        pygame.quit()

if __name__ == "__main__":
    app = OpenGLRenderer()
    app.run()