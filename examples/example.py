from pylygon import Polygon
import pygame
from pygame import display, draw, event, mouse, Surface
from pygame.font import Font
from pygame.locals import *


if __name__ == '__main__':
    pygame.init()

    SCREEN_SIZE = (800, 600)                                            #initialize screen size
    SCREEN = display.set_mode(SCREEN_SIZE)                              #load screen

    triangle = Polygon([(0, 70), (110, 0), (110, 70)])
    rhombus = Polygon([(0, 80), (20, 0), (80, 0), (60, 80)])

    triangle.move_ip(200, 200)
    rhombus.move_ip(300, 300)

    font = Font(None, 24)
    msg = font.render('collision!', 0, (255, 255, 255))

    grab = None
    while 1:
        SCREEN.fill((0, 0, 0))
        draw.polygon(SCREEN, (255, 0, 0), triangle.P)
        draw.polygon(SCREEN, (0, 0, 255), rhombus.P)
        mouse_pos = mouse.get_pos()
        for ev in event.get():
            if ev.type == KEYDOWN:
                if ev.key == K_q: exit()
            if ev.type == MOUSEBUTTONDOWN:
                if grab: grab = None
                elif rhombus.collidepoint(mouse_pos): grab = rhombus
                elif triangle.collidepoint(mouse_pos): grab = triangle

        if grab: grab.C = mouse_pos

        Y_triangle = triangle.project((0, 1))
        Y_rhombus = rhombus.project((0, 1))
        draw.line(SCREEN, (255, 0, 0), (2, Y_triangle[0]), (2, Y_triangle[1]), 3)
        draw.line(SCREEN, (0, 0, 255), (7, Y_rhombus[0]), (7, Y_rhombus[1]), 3)

        X_triangle = triangle.project((1, 0))
        X_rhombus = rhombus.project((1, 0))
        draw.line(SCREEN, (255, 0, 0), (X_triangle[0], 2), (X_triangle[1], 2), 3)
        draw.line(SCREEN, (0, 0, 255), (X_rhombus[0], 7), (X_rhombus[1], 7), 3)

        draw.circle(SCREEN, (255, 255, 255), triangle.C, 5)
        draw.circle(SCREEN, (255, 255, 255), rhombus.C, 5)

        if rhombus.collidepoly(triangle): SCREEN.blit(msg, (20, 20))

        display.update()
