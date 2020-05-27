# This work has been submitted to ACM Multimedia 2020
import open3d as o3d
import numpy as np
from random import randrange
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import BallTree
from tsp_solver.greedy import solve_tsp
import cv2
import os

############################################# (1) Initialization 
# Read ply file
def init(filename):
	pcd = o3d.io.read_point_cloud(filename)
	geo_arr = np.asarray(pcd.points)
	rgb_arr = np.asarray(pcd.colors)*255
	point_num = len(geo_arr)
	minPt = [min(geo_arr[:,0]), min(geo_arr[:,1]), min(geo_arr[:,2])]
	maxPt = [max(geo_arr[:,0]), max(geo_arr[:,1]), max(geo_arr[:,2])]
	pc_width = (maxPt[0]-minPt[0])
	y_comp_arr = [int(np.round(0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2], 0)) for rgb in rgb_arr]
	return [geo_arr, rgb_arr, y_comp_arr, point_num, pc_width]

# Read off (simplified point cloud) file
def read_off(off_path, geo_arr, rgb_arr):
	pt_arr = []
	off_rgb_arr = []
	idx = 0
	with open(off_path) as ins:
		for line in ins:
			re2 = line.replace("\n", "").split(" ")
			if idx>1:
				pt = [float(val) for val in re2[0:3]]
				pt_arr.append(pt)
				off_rgb_arr.append([int(val) for val in re2[3:6]])
			idx = idx + 1
	off_pt_num = len(pt_arr)
	return [pt_arr, off_rgb_arr, off_pt_num]

# Supervoxel generation (Assign points of point cloud to their corresponding nearest seed points)
def assign_ply_to_off(off_geo_arr, geo_arr, vis_flag):
	off_pt_assign_dic = dict()

	features = []
	labels = []

	for i in range(len(off_geo_arr)):
		features.append(off_geo_arr[i])
		labels.append(i)
		off_pt_assign_dic[i] = []

	clf = KNeighborsClassifier(n_neighbors = 1, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(geo_arr)

	for t in range(0, len(pre_label)):
		seg = pre_label[t]
		off_pt_assign_dic[seg].append(t)

	if vis_flag:
		vis_sc_geo = []
		vis_sc_rgb = []
		for off_id in off_pt_assign_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for id in off_pt_assign_dic[off_id]:
				vis_sc_geo.append(geo_arr[id])
				vis_sc_rgb.append(rgb)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(vis_sc_geo)
		pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_sc_rgb)/255.0)
		o3d.visualization.draw_geometries([pcd])

	return off_pt_assign_dic


############################################# (2) 3D point cloud linearization (3D to 1D)
# Travelling salesman problem (TSP) distance matrix construction
def construct_tsp_distance_matrix(pt_arr, neighbour_num):
	neigh = NearestNeighbors(n_neighbors = min(neighbour_num, len(pt_arr)))
	neigh.fit(pt_arr)
	kng = neigh.kneighbors_graph(pt_arr)
	kng = kng.toarray()
	dist_m = graph_shortest_path(kng, method='auto', directed=False)
	return dist_m

def bsp_traversal(geo_arr):
	tot_num = len(geo_arr)
	X = np.array(geo_arr)
	tree = BallTree(X, leaf_size=1)
	center = np.mean(geo_arr, axis=0)
	dist1, ind1 = tree.query([center], k=tot_num)
	farthest_pt1 = geo_arr[ind1[0][-1]]
	dist2, ind2 = tree.query([farthest_pt1], k=tot_num)
	farthest_pt2 = geo_arr[ind2[0][-1]]

	pivot_left_pt = farthest_pt1

	partition_arr_idx = [list(range(tot_num))]
	flag = 1
	iteration = 0
	while flag:
		temp_partition_arr = []
		for t in range(0, len(partition_arr_idx)):
			seg_geo_idx_arr = partition_arr_idx[t]
			seg_geo_arr = [geo_arr[id] for id in seg_geo_idx_arr]
			sub_tot_num = len(seg_geo_arr)
			if sub_tot_num>1:
				X = np.array(seg_geo_arr)
				tree = BallTree(X, leaf_size=1)

				if t == 0:
					farthest_pt1 = pivot_left_pt
					dist2, ind2 = tree.query([farthest_pt1], k=sub_tot_num)
					farthest_pt2 = seg_geo_arr[ind2[0][-1]]
				else:
					dist, ind = tree.query([farthest_pt2], k=1)
					farthest_pt1 = seg_geo_arr[ind[0][0]]
					dist2, ind2 = tree.query([farthest_pt1], k=sub_tot_num)
					farthest_pt2 = seg_geo_arr[ind2[0][-1]]

				temp_pt_idx_arr1 = []
				temp_pt_idx_arr2 = []
				for i in range(0, sub_tot_num):
					pt = seg_geo_arr[i]
					vec1 = np.asarray(farthest_pt1) - np.asarray(pt)
					vec2 = np.asarray(farthest_pt2) - np.asarray(pt)
					
					if np.linalg.norm(vec1) < np.linalg.norm(vec2):
						temp_pt_idx_arr1.append(seg_geo_idx_arr[i])
					else:
						temp_pt_idx_arr2.append(seg_geo_idx_arr[i])

				if len(temp_pt_idx_arr1):
					temp_partition_arr.append(temp_pt_idx_arr1)

				if len(temp_pt_idx_arr2):
					temp_partition_arr.append(temp_pt_idx_arr2)
			else:
				temp_partition_arr.append(seg_geo_idx_arr)

		partition_arr_idx = temp_partition_arr

		flag = 0
		for seg_geo_idx_arr in partition_arr_idx:
			if len(seg_geo_idx_arr)>1:
				flag = 1
				break
		iteration = iteration + 1
		print("iteration: ", iteration)

	bsp_traversal_idx_arr = []
	for s in range(0, len(partition_arr_idx)):
		rec = partition_arr_idx[s]
		for idx in rec:
			bsp_traversal_idx_arr.append(idx)

	return bsp_traversal_idx_arr

def bsp_traversal_with_tsp(geo_arr, off_geo_arr, off_ply_assign_dic):
	bsp_traversal_with_tsp_idx_arr = []

	off_bsp_traversal_idx = bsp_traversal(off_geo_arr) # traverse seed points using BSP based traversal
	seed_pt_num = len(off_geo_arr)
	st_idx = 0
	end_idx = 0
	pre_last_pt = []
	bsp_traversal_with_tsp_idx_arr = []
	for t in range(len(off_bsp_traversal_idx)):
		print(t, seed_pt_num)
		idx = off_bsp_traversal_idx[t]
		nex_off_pt = off_geo_arr[idx]
		curr_cluster_geo = [geo_arr[id] for id in off_ply_assign_dic[idx]]
		curr_cluster_idx = off_ply_assign_dic[idx]
		X = np.array(curr_cluster_geo)
		tree = BallTree(X, leaf_size=1)

		if t == 0:
			dist, ind = tree.query([nex_off_pt], k=1)
			end_idx = ind[0][0]
			dist, ind = tree.query([nex_off_pt], k=len(curr_cluster_geo))
			st_idx = ind[0][-1]
			pre_last_pt = curr_cluster_geo[end_idx]
		elif t == len(off_bsp_traversal_idx)-1:
			dist, ind = tree.query([pre_last_pt], k=1)
			st_idx = ind[0][0]
			dist, ind = tree.query([curr_cluster_geo[st_idx]], k=len(curr_cluster_geo))
			end_idx = ind[0][-1]
		else:
			dist, ind = tree.query([pre_last_pt], k=1)
			st_idx = ind[0][0]
			dist, ind = tree.query([nex_off_pt], k=1)
			end_idx = ind[0][0]
			if st_idx == end_idx:
				if len(curr_cluster_geo)>1:
					dist, ind = tree.query([nex_off_pt], k=2)
					end_idx = ind[0][1]	
			pre_last_pt = curr_cluster_geo[end_idx]

		cluster_path = []
		if len(curr_cluster_geo)>1:
			dist_m = construct_tsp_distance_matrix(curr_cluster_geo, neighbour_num = 8)
			cluster_path = solve_tsp(dist_m, endpoints=(st_idx, end_idx))
		else:
			if len(curr_cluster_geo):
				cluster_path = [0]

		for idx in cluster_path:
			bsp_traversal_with_tsp_idx_arr.append(curr_cluster_idx[idx])

	return bsp_traversal_with_tsp_idx_arr

############################################# (3) Attribute image generation using hybrid space-filling pattern (1D to 2D)
def hori_snake_curve(blk_color_arr, hori_snake_b):
	blk_img = np.ones((hori_snake_b, hori_snake_b, 3), np.uint8)*0
	for t in range(len(blk_color_arr)):
		y_pos = int(t/hori_snake_b)
		x_pos = ((-1)**(y_pos%2))*(t%hori_snake_b) + (hori_snake_b-1)*(y_pos%2)
		blk_img[y_pos][x_pos] = blk_color_arr[t]
	return blk_img

def rot(n, x, y, rx, ry):
	if (ry == 0):
		if (rx == 1):
			x = n-1 - x
			y = n-1 - y
		t = x
		x = y
		y = t
	return [x,y]

def get_hilbert_pos(hilbert_b, hilbert_idx):
	rx = 0
	ry = 0
	t = hilbert_idx
	x = 0
	y = 0
	s = 1
	while s<hilbert_b:
		rx = 1 & int(t/2)
		ry = 1 & (int(t) ^ rx)
		[x,y] = rot(s, x, y, rx, ry)
		x = x + s * rx
		y = y + s * ry
		t = t/4
		s = s*2
	return [x, y]

def hybrid_space_filling(color_arr, mb_size, hori_snake_b):
	hilbert_b = int(mb_size/hori_snake_b) # mb_size = 2^n1, 	hori_snake_b = 2^n2,		hori_snake_b < mb_size
	tot_pt_num = len(color_arr)
	blk_num = int(np.ceil(tot_pt_num/(mb_size*mb_size)))
	sub_blk_num = int(np.ceil(tot_pt_num/(hori_snake_b*hori_snake_b)))
	attr_img = np.ones((mb_size, blk_num*mb_size, 3), np.uint8)*0

	for blk_idx in range(sub_blk_num):
		blk_color_arr = color_arr[blk_idx*hori_snake_b*hori_snake_b:(blk_idx+1)*hori_snake_b*hori_snake_b]
		blk_img = hori_snake_curve(blk_color_arr, hori_snake_b)
		[hil_x, hil_y] = get_hilbert_pos(hilbert_b, blk_idx%(hilbert_b*hilbert_b))
		attr_img[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b + int(blk_idx/(hilbert_b*hilbert_b))*mb_size:(hil_x+1)*hori_snake_b + int(blk_idx/(hilbert_b*hilbert_b))*mb_size] = blk_img
	return attr_img

############################################
def img_compression(attr_img, pt_num, mb_size, cmp_method, extra_bit):
	mode = 0
	if cmp_method == "jpg":
		mode = int(cv2.IMWRITE_JPEG_QUALITY)
	elif cmp_method == "webp":
		mode = int(cv2.IMWRITE_WEBP_QUALITY)
	quality_arr = [20, 50, 80, 90]
		
	blk_size = int(np.floor(16376.0/mb_size)*mb_size)
	
	img_h, img_w, c = attr_img.shape
	attr_img_yuv = cv2.cvtColor(attr_img, cv2.COLOR_BGR2YUV)
	attr_img_y = attr_img_yuv[:,:,0]
	
	bpp_arr = []
	psnr_arr = []
	size_arr = []
	diff_arr = []
	for quality in quality_arr:
		yuv_size = 0
		tot_diff = 0.0
		for i in range(0, int(np.ceil(img_w/blk_size))):
			compressed_img_path = 'img\\' + str(i) + '_' + str(quality) + "." + cmp_method
			sub_attr_img = attr_img[:, i*blk_size:(i+1)*blk_size]
			cv2.imwrite(compressed_img_path, sub_attr_img, [mode, quality])
			sub_attr_img_y = attr_img_y[:, i*blk_size:(i+1)*blk_size]
			sub_attr_img_yuv_size = os.stat(compressed_img_path).st_size
			yuv_size = yuv_size + sub_attr_img_yuv_size
			compressed_yuv_img = cv2.imread(compressed_img_path)
			compressed_yuv_img_y = cv2.cvtColor(compressed_yuv_img, cv2.COLOR_BGR2YUV)[:,:,0]
			for s in range(0, compressed_yuv_img_y.shape[0]):
				for t in range(0, compressed_yuv_img_y.shape[1]):
					dif = int(sub_attr_img_y[s][t]) - int(compressed_yuv_img_y[s][t])
					dif = dif*dif
					tot_diff = tot_diff + dif
		mse = tot_diff/pt_num
		psnr = 20*np.log10(255.0/np.sqrt(mse))
		bpp = yuv_size*8.0/pt_num
		if extra_bit:
			bpp = (yuv_size*8.0 + extra_bit)/pt_num
		bpp_arr.append(bpp)
		psnr_arr.append(psnr)
		diff_arr.append(tot_diff)
		size_arr.append(yuv_size)

	return [bpp_arr, psnr_arr, diff_arr, size_arr]

############################################
def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
	lR1 = np.log(R1)
	lR2 = np.log(R2)

	# rate method
	p1 = np.polyfit(PSNR1, lR1, 3)
	p2 = np.polyfit(PSNR2, lR2, 3)

	# integration interval
	min_int = max(min(PSNR1), min(PSNR2))
	max_int = min(max(PSNR1), max(PSNR2))

	# find integral
	if piecewise == 0:
		p_int1 = np.polyint(p1)
		p_int2 = np.polyint(p2)

		int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
		int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
	else:
		lin = np.linspace(min_int, max_int, num=100, retstep=True)
		interval = lin[1]
		samples = lin[0]
		v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), np.sort(lR1), samples)
		v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), np.sort(lR2), samples)
		# Calculate the integral using the trapezoid method on the samples.
		int1 = np.trapz(v1, dx=interval)
		int2 = np.trapz(v2, dx=interval)

	# find avg diff
	avg_exp_diff = (int2-int1)/(max_int-min_int)
	avg_diff = (np.exp(avg_exp_diff)-1)*100
	return avg_diff

def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
	lR1 = np.log(R1)
	lR2 = np.log(R2)

	p1 = np.polyfit(lR1, PSNR1, 3)
	p2 = np.polyfit(lR2, PSNR2, 3)

	# integration interval
	min_int = max(min(lR1), min(lR2))
	max_int = min(max(lR1), max(lR2))

	# find integral
	if piecewise == 0:
		p_int1 = np.polyint(p1)
		p_int2 = np.polyint(p2)

		int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
		int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
	else:
		lin = np.linspace(min_int, max_int, num=100, retstep=True)
		interval = lin[1]
		samples = lin[0]
		v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), np.sort(PSNR1), samples)
		v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), np.sort(PSNR2), samples)
		# Calculate the integral using the trapezoid method on the samples.
		int1 = np.trapz(v1, dx=interval)
		int2 = np.trapz(v2, dx=interval)

	# find avg diff
	avg_diff = (int2-int1)/(max_int-min_int)

	return avg_diff

if __name__ == '__main__':
	pc_id_arr = ["andrew9_frame0027", "David_frame0000", "ricardo9_frame0039", "phil9_frame0244", "sarah9_frame0018", "Staue_Klimt", "Egyptian_mask", "Shiva_00035", "Facade_00009", "House_without_roof_00057", "Frog_00067", "Arco_Valentino_Dense"]
	frame_id = pc_id_arr[0]

	ply_path = "ply/" + frame_id + ".ply"
	
	#Seed points
	sv_pt_num = 128 # Average point number of supervoxels, it can also be set to smaller one (e.g. 128 or 64)
	off_path = "LOD_off/" + frame_id + "_n"+ str(sv_pt_num) + ".off" #
	
	off_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(sv_pt_num) + ".off"
	ply_path = floder + frame_id + ".ply"

	[geo_arr, rgb_arr, y_comp_arr, point_num, pc_width] = init(ply_path)
	[off_geo_arr, off_rgb_arr, off_pt_num] = read_off(off_path, geo_arr, rgb_arr)

	# Supervoxel generation
	off_ply_assign_dic = assign_ply_to_off(off_geo_arr, geo_arr, vis_flag=False)

	# Binary space partition (BSP) based universal traversal (with/without tsp)
	# bsp_traversal_idx_arr = bsp_traversal(geo_arr) # This function can also be used to traversal the whole point cloud
	bsp_traversal_with_tsp_idx_arr = bsp_traversal_with_tsp(geo_arr, off_geo_arr, off_ply_assign_dic)

	bsp_traversal_color_arr = [rgb_arr[idx][::-1] for idx in bsp_traversal_with_tsp_idx_arr]

	sfc_based_attr_img = hybrid_space_filling(bsp_traversal_color_arr, mb_size = 16, hori_snake_b = 4)
	[bpp_arr, psnr_arr, diff_arr, size_arr] = img_compression(sfc_based_attr_img, len(bsp_traversal_color_arr), mb_size = 16, cmp_method = "jpg", extra_bit = 0) #cmp_method = "webp"
	print([bpp_arr, psnr_arr, diff_arr, size_arr])

  
  #to be continued
