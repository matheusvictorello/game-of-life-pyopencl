import pyopencl as cl
import numpy as np
import pygame
import timeit
import sys

BLOCK_SIDE = 8
BLOCKS_PER_DIM = 32

N = BLOCK_SIDE * BLOCKS_PER_DIM + 2

BLOCK = (BLOCK_SIDE, BLOCK_SIDE)

TYPE = np.int32
TYPE_SIZE = np.dtype(TYPE).itemsize

SQR_SIZE = 4
WIDTH, HEIGHT = N*SQR_SIZE, N*SQR_SIZE

SEE_ACTIVE_BLOCKS = True

STEP_SIM = 0
WARP = 0
GEN_TO_WARP = 0
FPS = 0

kernel = f"""
	inline int idx(int a, int b) {{
		return a*{N} + b;
	}}

	__kernel void gameStep(
		__global int *m,
		__global int *n,
		__global int *flag)
	{{
		int x = get_global_id(0);
		int y = get_global_id(1);

		int a00 = m[idx(x-1, y-1)];
		int a01 = m[idx(x-1, y  )];
		int a02 = m[idx(x-1, y+1)];
		
		int a10 = m[idx(x  , y-1)];
		int a11 = m[idx(x  , y  )];
		int a12 = m[idx(x  , y+1)];
		
		int a20 = m[idx(x+1, y-1)];
		int a21 = m[idx(x+1, y  )];
		int a22 = m[idx(x+1, y+1)];

		int s = a00 + a01 + a02
			  + a10       + a12
			  + a20 + a21 + a22;

		if (s < 2) {{
			n[idx(x, y)] = 0;
		}} else if (s == 2) {{
			n[idx(x, y)] = a11;
			
			if (a11) {{
				flag[idx(0, 0)] = 1;
			}}
		}} else if (s == 3) {{
			n[idx(x, y)] = 1;
			flag[idx(0, 0)] = 1;
		}} else if (s > 3) {{
			n[idx(x, y)] = 0;
		}}
	}}
"""

def main():
	pygame.init()

	steps = 0
	active_blocks = []

	context = cl.create_some_context()
	queue = cl.CommandQueue(context)
	program = cl.Program(context, kernel).build()
	# device = queue.get_info(cl.command_queue_info.DEVICE)
	# wg = program.f.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)

	mf = cl.mem_flags
	field_0 = cl.Buffer(context, mf.READ_WRITE, N*N*TYPE_SIZE)
	field_1 = cl.Buffer(context, mf.READ_WRITE, N*N*TYPE_SIZE)
	flag = cl.Buffer(context, mf.READ_WRITE, TYPE_SIZE)

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

		ret_flag = np.zeros((1, 1)).astype(TYPE)

		cl.enqueue_copy(queue, flag, ret_flag)

		program.gameStep(queue, BLOCK, None, fa, fb, flag, global_offset=(1 + block_ix*BLOCK_SIDE, 1 + block_iy*BLOCK_SIDE))

		cl.enqueue_copy(queue, ret_flag, flag)
		
		return ret_flag[0][0]

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

		offset = px*N + py

		if steps%2 == 0:
			for i, line in enumerate(pattern):
				cl.enqueue_copy(queue, field_0, line, device_offset=offset*TYPE_SIZE)
				offset += N
		else:
			for i, line in enumerate(pattern):
				cl.enqueue_copy(queue, field_1, line, device_offset=offset*TYPE_SIZE)
				offset += N

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

		ret = np.zeros((BLOCK_SIDE, BLOCK_SIDE)).astype(TYPE)
		
		if buff == None:
			if steps%2 == 0:
				buff = field_0
			else:
				buff = field_1

		bx *= BLOCK_SIDE
		by *= BLOCK_SIDE

		offset = bx*N + by + N + 1

		for line in ret:
			cl.enqueue_copy(queue, line, buff, device_offset=offset*TYPE_SIZE)
			offset += N

		return ret

	def make_pattern(list):
		return np.array(list).astype(TYPE)

	def draw(screen, m, dx, dy):
		for i, line in enumerate(m):
			for j, cell in enumerate(line):
				if cell == 1:
					pygame.draw.rect(screen, (255, 255, 255), (j*SQR_SIZE+dy*SQR_SIZE*BLOCK_SIDE+SQR_SIZE, i*SQR_SIZE+dx*SQR_SIZE*BLOCK_SIDE+SQR_SIZE, SQR_SIZE, SQR_SIZE))

	def warp(n):
		print(timeit.timeit(full_step, number=n))

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

	running = False
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()

	if WARP:
		warp(GEN_TO_WARP)

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

		if STEP == 0:
			full_step()

		if not FPS == 0:
			clock.tick(FPS)

if __name__ == '__main__':
	args = sys.argv[1:]

	if '--step' in args:
		STEP_SIM = 1

	if '--warp' in args:
		WARP = 1

		idx = args.index('--warp') + 1
		try:
			n = args + 1
		except:
			print('--warp <n>')
			exit()

		GEN_TO_WARP = int(args[n])

	if '--fps' in args:
		idx = args.index('--warp') + 1
		try:
			n = args + 1
		except:
			print('--fps <n>')
			exit()

		FPS = int(args[n])

	main()