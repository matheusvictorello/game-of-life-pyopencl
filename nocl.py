import numpy as np
import pygame
import timeit

BLOCK_SIDE = 8
BLOCKS_PER_DIM = 32

N = BLOCK_SIDE * BLOCKS_PER_DIM + 2

BLOCK = (BLOCK_SIDE, BLOCK_SIDE)

TYPE = np.int32
TYPE_SIZE = np.dtype(TYPE).itemsize

pygame.init()

SQR_SIZE = 4
WIDTH, HEIGHT = N*SQR_SIZE, N*SQR_SIZE

SEE_ACTIVE_BLOCKS = True

def main():
	steps = 0
	active_blocks = []

	field_0 = np.zeros((N, N)).astype(TYPE)
	field_1 = np.zeros((N, N)).astype(TYPE)

	def block_is_active(block_idx):
		x, y = block_idx
		x, y = x - 1, y - 1

		candidates = []

		for i in range(x, x+3):
			for j in range(y, y+3):
				if i >= 0 and j >= 0 and i < BLOCKS_PER_DIM and j < BLOCKS_PER_DIM:
					candidates.append((i, j))

		for c in candidates:
			if not c in active_blocks:
				active_blocks.append(c)

	def flip():
		nonlocal steps
		steps += 1

	def step(block_ix, block_iy):
		nonlocal steps

		if steps%2 == 0:
			fa = field_0
			fb = field_1
		else:
			fa = field_1
			fb = field_0

		ret_flag = 0

		# gameStep

		x = 1+block_ix*BLOCK_SIDE
		y = 1+block_iy*BLOCK_SIDE
		
		for i in range(x, x+BLOCK_SIDE):
			for j in range(y, y+BLOCK_SIDE):
				a00 = fa[i-1][j-1]
				a01 = fa[i-1][j]
				a02 = fa[i-1][j+1]
				
				a10 = fa[i][j-1]
				a11 = fa[i][j]
				a12 = fa[i][j+1]
				
				a20 = fa[i+1][j-1]
				a21 = fa[i+1][j]
				a22 = fa[i+1][j+1]

				s = a00 + a01 + a02 + a10 + a12 + a20 + a21 + a22

				if (s < 2):
					fb[i][j] = 0
				elif (s == 2):
					fb[i][j] = a11
					
					if (a11):
						flag = 1
				elif (s == 3):
					fb[i][j] = 1
					ret_flag = 1
				elif (s > 3):
					fb[i][j] = 0

		return ret_flag

	def full_step():
		nonlocal active_blocks

		updating = active_blocks
		active_blocks = []

		for b_idx in updating:
			bx, by = b_idx
			is_active = step(bx, by)

			if is_active:
				block_is_active(b_idx)

		flip()

	def plot(pattern, px, py):
		nonlocal steps

		sx, sy = pattern.shape

		if steps%2 == 0:
			field_0[px:px+sx, py:py+sy] = pattern

		else:
			field_0[px:px+sx, py:py+sy] = pattern

		# Block activation
		dx, dy = pattern.shape
		dx, dy = dx - 1, dy - 1

		x1, y1 = (px+1)//BLOCK_SIDE,    (py+1)//BLOCK_SIDE
		x2, y2 = (px+dx+1)//BLOCK_SIDE, (py+dy+1)//BLOCK_SIDE

		for i in range(x1, x2+1):
			for j in range(y1, y2+1):
				block_is_active((i, j))

	def see(bx, by, buff=None):
		# Usar depois do flip
		
		if buff == None:
			if steps%2 == 0:
				buff = field_0
			else:
				buff = field_1

		bx *= BLOCK_SIDE+1
		by *= BLOCK_SIDE+1

		return buff[bx:bx+BLOCK_SIDE, by:by+BLOCK_SIDE]

	def make_pattern(list):
		return np.array(list).astype(TYPE)

	def draw(screen, m, dx, dy):
		for i, line in enumerate(m):
			for j, cell in enumerate(line):
				if cell == 1:
					pygame.draw.rect(screen, (255, 255, 255), (j*SQR_SIZE+dy*SQR_SIZE*BLOCK_SIDE+SQR_SIZE, i*SQR_SIZE+dx*SQR_SIZE*BLOCK_SIDE+SQR_SIZE, SQR_SIZE, SQR_SIZE))

	def warp(n):
		for _ in range(n):
			full_step()

	# LR
	glider = make_pattern([
		[0,0,1],
		[1,0,1],
		[0,1,1]
	])

	# LR
	gunner = make_pattern([
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
		[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
		[1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	])

	bloco = make_pattern([
		[1,1],
		[1,1]
	])

	bote = make_pattern([
		[1,1,0],
		[1,0,1],
		[0,1,0]
	])

	blinker = make_pattern([
		[1,1,1]
	])

	sapo = make_pattern([
		[0,1,1,1],
		[1,1,1,0]
	])

	lwss = make_pattern([
		[0,1,0,0,1],
		[1,0,0,0,0],
		[1,0,0,0,1],
		[1,1,1,1,0]
	])

	diehard = make_pattern([
		[0,0,0,0,0,0,1,0],
		[1,1,0,0,0,0,0,0],
		[0,1,0,0,0,1,1,1]
	])

	acorn = make_pattern([
		[0,1,0,0,0,0,0,],
		[0,0,0,1,0,0,0,],
		[1,1,0,0,1,1,1,]
	])

	plot(gunner, 70, 50)

	# Loop

	running = True
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()

	# warp(1000)
	print(timeit.timeit(full_step, number=1000))

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				full_step()

		screen.fill((0, 0, 0))

		for i, j in active_blocks:
			if SEE_ACTIVE_BLOCKS:
				if (i+j)%2==0:
					color = (255, 0, 0)
				else:
					color = (0, 255, 0)

				pygame.draw.rect(screen, color, (j*SQR_SIZE*BLOCK_SIDE+SQR_SIZE, i*SQR_SIZE*BLOCK_SIDE+SQR_SIZE, SQR_SIZE*BLOCK_SIDE, SQR_SIZE*BLOCK_SIDE))

			ret = see(i, j)

			draw(screen, ret, i, j)

		pygame.display.flip()

		full_step()

		# clock.tick(20)

if __name__ == '__main__':
	main()