# This work has been submitted to ACM Multimedia 2020

import open3d as o3d
import numpy as np
from random import randrange
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import BallTree
from sklearn import manifold
from tsp_solver.greedy import solve_tsp
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import rasterfairy

##############################################################################################################################
# 																						 									 #
#											(1) Initialization and Preprocessing											 #
# 																						 									 #
##############################################################################################################################
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



##############################################################################################################################
# 																						 									 #
# 							(2) Scheme I: SFC-based Point Cloud Attribuate Image Generation 								 #
# 																						 									 #
##############################################################################################################################
### BSP-based 3D point cloud linearization (3D to 1D)

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

### Attribute image generation using hybrid space-filling pattern (1D to 2D)
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


##############################################################################################################################
# 																						 									 #
# 							(3) Scheme II: IsoMap-based Point Cloud Attribuate Image Generation 							 #
# 																						 									 #
##############################################################################################################################
def get_neighbor_dis(pt_arr):
	tot_num = len(pt_arr)
	X = np.array(pt_arr)
	tree = BallTree(X, leaf_size = 1)
	neighbor_num = 2
	dis_arr = []
	for i in range(0, tot_num, max(int(tot_num/1000), 1)):
		dist, ind = tree.query([pt_arr[0]], k=neighbor_num)
		dis_arr.append(dist[0][1])
	neighbor_dis = min(dis_arr)
	return neighbor_dis

def dbscan_clustering(pt_arr, dbscan_thresh):
	clustering = DBSCAN(eps=dbscan_thresh, min_samples=1).fit(pt_arr)
	label_arr = list(clustering.labels_)
	cluster_dic = dict()
	for i in range(len(pt_arr)):
		label = label_arr[i]
		if not label in cluster_dic:
			cluster_dic[label] = []
		cluster_dic[label].append(i)
	return cluster_dic

def reshape_sfc_based_attribute_img(patch_rgb, square_w, mb_size, hori_snake_b):
	patch_sfc_img = hybrid_space_filling([rgb[::-1] for rgb in patch_rgb], mb_size, hori_snake_b)
	patch_sfc_img_squre = np.ones((square_w, square_w, 3), np.uint8)*0 # reshape sfc-based attribute image by stacking its fragments vertically. 

	for s in range(0, int(square_w/mb_size)):
		sub_img = patch_sfc_img[:, s*square_w:(s+1)*square_w]
		sub_img_h, sub_img_w, c = sub_img.shape
		if s%2 == 1:
			for ss in range(0, sub_img_w):
				patch_sfc_img_squre[s*mb_size:(s+1)*mb_size, square_w - ss - 1] = sub_img[:, ss]
		else:
			for ss in range(0, sub_img_w):
				patch_sfc_img_squre[s*mb_size:(s+1)*mb_size, ss] = sub_img[:, ss]

	return patch_sfc_img_squre

def isomap_based_dimension_reduction(patch_geo_arr, patch_off_arr, landmark_flag):
	if landmark_flag:
		n_neighbors = min(8, len(patch_off_arr))
		n_components = 2
		X_off8 = np.matrix(patch_off_arr)
		embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		Y_off8 = embedding.fit_transform(X_off8)
		reconstruction_err = embedding.reconstruction_error()
		Y = []
		if len(patch_geo_arr)>5000:
			for t in range(0, len(patch_geo_arr), 5000):
				print(t, len(patch_geo_arr))
				sub_non_smooth_sc_geo_arr = patch_geo_arr[t:t+5000]
				sub_X = np.matrix(sub_non_smooth_sc_geo_arr)
				sub_Y = embedding.transform(sub_X)
				for val in sub_Y:
					Y.append(val)
		else:
			X = np.matrix(patch_geo_arr)
			Y = embedding.transform(X)
		d2_geo = [[val[0], val[1], 0] for val in Y]

		return [d2_geo, reconstruction_err]
	else:
		n_neighbors = min(8, len(patch_geo_arr)-1)
		n_components = 2
		embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		X = np.matrix(patch_geo_arr)
		Y = embedding.fit_transform(X)
		reconstruction_err = embedding.reconstruction_error()
		d2_geo = [[val[0], val[1], 0] for val in Y]

		return [d2_geo, reconstruction_err]

def placement_cost(img):
	img_h, img_w, c = img.shape
	cost = 0

	for i in range(img_h):
		for j in range(img_w):
			[b, g, r] = img[i][j]
			cur_y = int(np.round(0.299*r + 0.587*g + 0.114*b, 0))
			try:
				[b, g, r] = img[i+1][j]
				nei_y = int(np.round(0.299*r + 0.587*g + 0.114*b, 0))
				cost = cost + np.abs(nei_y-cur_y)
			except:
				continue
			try:
				[b, g, r] = img[i-1][j]
				nei_y = int(np.round(0.299*r + 0.587*g + 0.114*b, 0))
				cost = cost + np.abs(nei_y-cur_y)
			except:
				continue
			try:
				[b, g, r] = img[i][j+1]
				nei_y = int(np.round(0.299*r + 0.587*g + 0.114*b, 0))
				cost = cost + np.abs(nei_y-cur_y)
			except:
				continue
			try:
				[b, g, r] = img[i][j-1]
				nei_y = int(np.round(0.299*r + 0.587*g + 0.114*b, 0))
				cost = cost + np.abs(nei_y-cur_y)
			except:
				continue
	return cost

def isomap_based_attr_img_gen(traversal_idx_arr, geo_arr, rgb_arr, mb_size, hori_snake_b):
	tot_pt_num = len(traversal_idx_arr)
	square_w = 64 # size of patch attribute image mb_size
	patch_size = square_w*square_w # number of points of a patch

	ply_neighbor_dis = get_neighbor_dis(geo_arr)
	dbscan_thresh = ply_neighbor_dis*5
	isomap_thresh = 20

	pc_attr_img = np.ones((square_w, 0, 3), np.uint8)*0

	for t in range(0, tot_pt_num, patch_size):
		patch_geo = []
		patch_rgb = []
		patch_off_geo_arr = [] # the simplified point cloud of a patch can be used to improve the IsoMap-based embedding efficiency 
		for idx in traversal_idx_arr[t:t+patch_size]:
			patch_geo.append(geo_arr[idx])
			patch_rgb.append(rgb_arr[idx])
			# patch_off_geo_arr.append()

		patch_sfc_img_squre = reshape_sfc_based_attribute_img(patch_rgb, square_w, mb_size, hori_snake_b)
		sfc_cost = placement_cost(patch_sfc_img_squre)

		y_comp_arr = [int(np.round(0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2], 0)) for rgb in patch_rgb]
		cluster_y_std = np.std(y_comp_arr)

		# if cluster_y_std < lower_threshold or cluster_y_std > upper_threshold: # For patches with very high or very low color variance, SFC-based scheme generally can work very well. 
		if t + patch_size >= tot_pt_num - 1 and tot_pt_num%patch_size != 0:
			pc_attr_img = np.hstack((pc_attr_img, patch_sfc_img_squre))
		else:
			cluster_dic = dbscan_clustering(patch_geo, dbscan_thresh)
			if len(cluster_dic) == 1:
				[patch_geo_arr_d2, recon_err] = isomap_based_dimension_reduction(patch_geo, patch_off_geo_arr, len(patch_off_geo_arr)>8)
				print(t/patch_size, recon_err)
				if recon_err > isomap_thresh:
					pc_attr_img = np.hstack((pc_attr_img, patch_sfc_img_squre))
				else:
					grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(patch_geo_arr_d2)[:, 0:2], autoAdjustCount = False)
					grid_img = np.ones((height, width, 3), np.uint8)*0

					for i in range(0, len(patch_geo_arr_d2)):
						grid_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = patch_rgb[i][::-1]

					isomap_cost = placement_cost(grid_img)
					if sfc_cost > isomap_cost:
						pc_attr_img = np.hstack((pc_attr_img, grid_img))
					else:
						pc_attr_img = np.hstack((pc_attr_img, patch_sfc_img_squre))

			else:		
				d2_dic = dict()
				mode_flag = 1
				for cluster_id in cluster_dic:
					sub_patch_geo = []
					sub_patch_rgb = []
					sub_patch_off_geo_arr = []
					for idx in cluster_dic[cluster_id]:
						sub_patch_geo.append(patch_geo[idx])
						sub_patch_rgb.append(patch_rgb[idx])
					[sub_patch_geo_arr_d2, sub_recon_err] = isomap_based_dimension_reduction(sub_patch_geo, sub_patch_off_geo_arr, len(sub_patch_off_geo_arr)>8)
					print(t/patch_size, sub_recon_err)
					d2_dic[cluster_id] = [sub_patch_geo_arr_d2, sub_patch_rgb, len(sub_patch_rgb)]
					if sub_recon_err> isomap_thresh:
						mode_flag = 0
						break

				if mode_flag:
					sorted_x = sorted(d2_dic.items(), key=lambda x: x[1][2], reverse=True)
					merged_patch_geo_d2 = []
					merged_patch_rgb = []
					for rec in sorted_x:
						sub_patch_geo_arr_d2 = rec[1][0]
						sub_patch_rgb = rec[1][1]
						if len(merged_patch_geo_d2) == 0:
							merged_patch_geo_d2 = merged_patch_geo_d2 + sub_patch_geo_arr_d2
							merged_patch_rgb = merged_patch_rgb + sub_patch_rgb
						else:
							max_x = np.max(np.asarray(merged_patch_geo_d2)[:, 0])
							max_y = np.max(np.asarray(merged_patch_geo_d2)[:, 1])
							sub_min_x = np.min(np.asarray(sub_patch_geo_arr_d2)[:, 0])
							sub_min_y = np.min(np.asarray(sub_patch_geo_arr_d2)[:, 1])
							mov_x = max_x - sub_min_x
							mov_y = max_y - sub_min_y
							merged_patch_rgb = merged_patch_rgb + sub_patch_rgb
							for pt in sub_patch_geo_arr_d2:
								merged_patch_geo_d2.append([pt[0] + mov_x*1.2, pt[1] + mov_y*1.2, pt[2]])

					grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(merged_patch_geo_d2)[:, 0:2], autoAdjustCount = False)
					grid_img = np.ones((height, width, 3), np.uint8)*0

					for i in range(0, len(merged_patch_geo_d2)):
						grid_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = merged_patch_rgb[i][::-1]

					isomap_cost = placement_cost(grid_img)

					if sfc_cost > isomap_cost:
						pc_attr_img = np.hstack((pc_attr_img, grid_img))
					else:
						pc_attr_img = np.hstack((pc_attr_img, patch_sfc_img_squre))
				else:
					pc_attr_img = np.hstack((pc_attr_img, patch_sfc_img_squre))

	return pc_attr_img
			
##############################################################################################################################
# 																						 									 #
# 												(4) Image compression 														 #
# 																						 									 #
############################################################################################################################## 
def img_compression(attr_img, pt_num, mb_size, cmp_method, extra_bit):
	mode = 0
	if cmp_method == "jpg":
		mode = int(cv2.IMWRITE_JPEG_QUALITY)
	elif cmp_method == "webp":
		mode = int(cv2.IMWRITE_WEBP_QUALITY)
	quality_arr = [20, 50, 80, 90]
		
	blk_size = int(np.floor(16376.0/mb_size)*mb_size) # The maximum pixel dimensions of a WebP image is 16383 x 16383.
	
	img_h, img_w, c = attr_img.shape
	attr_img_yuv = cv2.cvtColor(attr_img, cv2.COLOR_BGR2YUV)
	attr_img_y = attr_img_yuv[:,:,0]
	
	bpp_arr = []
	psnr_arr = []
	# size_arr = []
	# diff_arr = []
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
		# diff_arr.append(tot_diff)
		# size_arr.append(yuv_size)

	return [bpp_arr, psnr_arr]


##############################################################################################################################
# 																						 									 #
# 												(5) Other functionalites 													 #
# 																						 									 #
##############################################################################################################################
### Octree-based (depth first traversal) 3D point cloud linearization (3D to 1D)
class OctNode(object):
	"""
	New Octnode Class, can be appended to as well i think
	"""
	def __init__(self, position, size, depth, data):
		"""
		OctNode Cubes have a position and size
		position is related to, but not the same as the objects the node contains.

		Branches (or children) follow a predictable pattern to make accesses simple.
		Here, - means less than 'origin' in that dimension, + means greater than.
		branch: 0 1 2 3 4 5 6 7
		x:      - - - - + + + +
		y:      - - + + - - + +
		z:      - + - + - + - +
		"""
		self.position = position
		self.size = size
		self.depth = depth

		## All OctNodes will be leaf nodes at first
		## Then subdivided later as more objects get added
		self.isLeafNode = True

		## store our object, typically this will be one, but maybe more
		self.data = data

		## might as well give it some emtpy branches while we are here.
		self.branches = [None, None, None, None, None, None, None, None]

		half = size / 2

		## The cube's bounding coordinates
		self.lower = (position[0] - half, position[1] - half, position[2] - half)
		self.upper = (position[0] + half, position[1] + half, position[2] + half)

	def __str__(self):
		data_str = u", ".join((str(x) for x in self.data))
		return u"position: {0}, size: {1}, depth: {2} leaf: {3}, data: {4}".format(
			self.position, self.size, self.depth, self.isLeafNode, data_str
		)

class Octree(object):
	"""
	The octree itself, which is capable of adding and searching for nodes.
	"""
	def __init__(self, worldSize, origin=(0, 0, 0), max_type="nodes", max_value=10):
		"""
		Init the world bounding root cube
		all world geometry is inside this
		it will first be created as a leaf node (ie, without branches)
		this is because it has no objects, which is less than MAX_OBJECTS_PER_CUBE
		if we insert more objects into it than MAX_OBJECTS_PER_CUBE, then it will subdivide itself.

		"""
		self.root = OctNode(origin, worldSize, 0, [])
		self.worldSize = worldSize
		self.limit_nodes = (max_type=="nodes")
		self.limit = max_value

	@staticmethod
	def CreateNode(position, size, objects):
		"""This creates the actual OctNode itself."""
		return OctNode(position, size, objects)

	def insertNode(self, position, objData=None):
		"""
		Add the given object to the octree if possible

		Parameters
		----------
		position : array_like with 3 elements
			The spatial location for the object
		objData : optional
			The data to store at this position. By default stores the position.

			If the object does not have a position attribute, the object
			itself is assumed to be the position.

		Returns
		-------
		node : OctNode or None
			The node in which the data is stored or None if outside the
			octree's boundary volume.

		"""
		if np:
			if np.any(position < self.root.lower):
				return None
			if np.any(position > self.root.upper):
				return None
		else:
			if position < self.root.lower:
				return None
			if position > self.root.upper:
				return None

		if objData is None:
			objData = position

		return self.__insertNode(self.root, self.root.size, self.root, position, objData)

	def __insertNode(self, root, size, parent, position, objData):
		"""Private version of insertNode() that is called recursively"""
		if root is None:
			# we're inserting a single object, so if we reach an empty node, insert it here
			# Our new node will be a leaf with one object, our object
			# More may be added later, or the node maybe subdivided if too many are added
			# Find the Real Geometric centre point of our new node:
			# Found from the position of the parent node supplied in the arguments
			pos = parent.position

			## offset is halfway across the size allocated for this node
			offset = size / 2

			## find out which direction we're heading in
			branch = self.__findBranch(parent, position)

			## new center = parent position + (branch direction * offset)
			newCenter = (0, 0, 0)

			if branch == 0:
				newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset )
			elif branch == 1:
				newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset )
			elif branch == 2:
				newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset )
			elif branch == 3:
				newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset )
			elif branch == 4:
				newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset )
			elif branch == 5:
				newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset )
			elif branch == 6:
				newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset )
			elif branch == 7:
				newCenter = (pos[0] + offset, pos[1] + offset, pos[2] + offset )

			# Now we know the centre point of the new node
			# we already know the size as supplied by the parent node
			# So create a new node at this position in the tree
			# print "Adding Node of size: " + str(size / 2) + " at " + str(newCenter)
			return OctNode(newCenter, size, parent.depth + 1, [objData])

		#else: are we not at our position, but not at a leaf node either
		elif (
			not root.isLeafNode
			and
			(
				(np and np.any(root.position != position))
				or
				(root.position != position)
			)
		):

			# we're in an octNode still, we need to traverse further
			branch = self.__findBranch(root, position)
			# Find the new scale we working with
			newSize = root.size / 2
			# Perform the same operation on the appropriate branch recursively
			root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, position, objData)

		# else, is this node a leaf node with objects already in it?
		elif root.isLeafNode:
			# We've reached a leaf node. This has no branches yet, but does hold
			# some objects, at the moment, this has to be less objects than MAX_OBJECTS_PER_CUBE
			# otherwise this would not be a leafNode (elementary my dear watson).
			# if we add the node to this branch will we be over the limit?
			if (
				(self.limit_nodes and len(root.data) < self.limit)
				or
				(not self.limit_nodes and root.depth >= self.limit)
			):
				# No? then Add to the Node's list of objects and we're done
				root.data.append(objData)
				#return root
			else:
				# Adding this object to this leaf takes us over the limit
				# So we have to subdivide the leaf and redistribute the objects
				# on the new children.
				# Add the new object to pre-existing list
				root.data.append(objData)
				# copy the list
				objList = root.data
				# Clear this node's data
				root.data = None
				# It is not a leaf node anymore
				root.isLeafNode = False
				# Calculate the size of the new children
				newSize = root.size / 2
				# distribute the objects on the new tree
				# print "Subdividing Node sized at: " + str(root.size) + " at " + str(root.position)
				for ob in objList:
					# Use the position attribute of the object if possible
					if hasattr(ob, "position"):
						pos = ob.position
					else:
						pos = ob
					branch = self.__findBranch(root, pos)
					root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, pos, ob)
		return root

	def findPosition(self, position):
		"""
		Basic lookup that finds the leaf node containing the specified position
		Returns the child objects of the leaf, or None if the leaf is empty or none
		"""
		if np:
			if np.any(position < self.root.lower):
				return None
			if np.any(position > self.root.upper):
				return None
		else:
			if position < self.root.lower:
				return None
			if position > self.root.upper:
				return None
		return self.__findPosition(self.root, position)

	@staticmethod
	def __findPosition(node, position, count=0, branch=0):
		"""Private version of findPosition """
		if node.isLeafNode:
			#print("The position is", position, " data is", node.data)
			return node.data
		branch = Octree.__findBranch(node, position)
		child = node.branches[branch]
		if child is None:
			return None
		return Octree.__findPosition(child, position, count + 1, branch)

	@staticmethod
	def __findBranch(root, position):
		"""
		helper function
		returns an index corresponding to a branch
		pointing in the direction we want to go
		"""
		index = 0
		if (position[0] >= root.position[0]):
			index |= 4
		if (position[1] >= root.position[1]):
			index |= 2
		if (position[2] >= root.position[2]):
			index |= 1
		return index

	def iterateDepthFirst(self):
		"""Iterate through the octree depth-first"""
		gen = self.__iterateDepthFirst(self.root)
		for n in gen:
			yield n

	@staticmethod
	def __iterateDepthFirst(root):
		"""Private (static) version of iterateDepthFirst"""

		for branch in root.branches:
			if branch is None:
				continue
			for n in Octree.__iterateDepthFirst(branch):
				yield n
			if branch.isLeafNode:
				yield branch

def octree_depth_first_traversal(geo_arr): 
	class TestObject(object):
		def __init__(self, name, position):
			self.name = name
			self.position = position

		def __str__(self):
			return u"name: {0} position: {1}".format(self.name, self.position)
	NUM_TEST_OBJECTS = len(geo_arr)

	testObjects = []
	for x in range(NUM_TEST_OBJECTS):
		the_name = "Node__" + str(x)
		the_pos = (
			geo_arr[x][0],
			geo_arr[x][1],
			geo_arr[x][2]
		)
		testObjects.append(TestObject(the_name, the_pos))

	test_trees = (
		("nodes", 1),
		("depth", 12)
	)

	WORLD_SIZE = 2000

	# ORIGIN = (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE)
	ORIGIN = (500, 500, 500)
	octree_dft_idx_arr = []
	for tree_params in test_trees:
		myTree = Octree(
			WORLD_SIZE,
			ORIGIN,
			max_type=tree_params[0],
			max_value=tree_params[1]
		)

		# Start = time.time()
		for testObject in testObjects:
			myTree.insertNode(testObject.position, testObject)
		# End = time.time() - Start

		# if myTree.limit_nodes:
		#     print("Tree Leaves contain a maximum of", myTree.limit, "objects each.")
		# else:
		#     print("Tree has a maximum depth of", myTree.limit)

		# print("Depth First")
		
		for i, x in enumerate(myTree.iterateDepthFirst()):
			# print(i, ":", x)
			pt_id = str(x).split("name: Node__")[-1].split(" position")[0]
			# if not int(pt_id) in octree_dft_idx_arr:
			octree_dft_idx_arr.append(int(pt_id))
		break
	return octree_dft_idx_arr

### 3D Hilbert space-filling curve based 3D point cloud linearization (3D to 1D)
def _binary_repr(num, width):
	"""Return a binary string representation of `num` zero padded to `width`
	bits."""
	return format(num, 'b').zfill(width)

class HilbertCurve:

	def __init__(self, p, n):
		"""Initialize a hilbert curve with,

		Args:
			p (int): iterations to use in the hilbert curve
			n (int): number of dimensions
		"""
		if p <= 0:
			raise ValueError('p must be > 0')
		if n <= 0:
			raise ValueError('n must be > 0')
		self.p = p
		self.n = n

		# maximum distance along curve
		self.max_h = 2**(self.p * self.n) - 1

		# maximum coordinate value in any dimension
		self.max_x = 2**self.p - 1

	def _hilbert_integer_to_transpose(self, h):
		"""Store a hilbert integer (`h`) as its transpose (`x`).

		Args:
			h (int): integer distance along hilbert curve

		Returns:
			x (list): transpose of h
					  (n components with values between 0 and 2**p-1)
		"""
		h_bit_str = _binary_repr(h, self.p*self.n)
		x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
		return x

	def _transpose_to_hilbert_integer(self, x):
		"""Restore a hilbert integer (`h`) from its transpose (`x`).

		Args:
			x (list): transpose of h
					  (n components with values between 0 and 2**p-1)

		Returns:
			h (int): integer distance along hilbert curve
		"""
		x_bit_str = [_binary_repr(x[i], self.p) for i in range(self.n)]
		h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
		return h

	def coordinates_from_distance(self, h):
		"""Return the coordinates for a given hilbert distance.

		Args:
			h (int): integer distance along hilbert curve

		Returns:
			x (list): transpose of h
					  (n components with values between 0 and 2**p-1)
		"""
		if h > self.max_h:
			raise ValueError('h={} is greater than 2**(p*N)-1={}'.format(h, self.max_h))
		if h < 0:
			raise ValueError('h={} but must be > 0'.format(h))

		x = self._hilbert_integer_to_transpose(h)
		Z = 2 << (self.p-1)

		# Gray decode by H ^ (H/2)
		t = x[self.n-1] >> 1
		for i in range(self.n-1, 0, -1):
			x[i] ^= x[i-1]
		x[0] ^= t

		# Undo excess work
		Q = 2
		while Q != Z:
			P = Q - 1
			for i in range(self.n-1, -1, -1):
				if x[i] & Q:
					# invert
					x[0] ^= P
				else:
					# exchange
					t = (x[0] ^ x[i]) & P
					x[0] ^= t
					x[i] ^= t
			Q <<= 1

		# done
		return x

	def distance_from_coordinates(self, x_in):
		"""Return the hilbert distance for a given set of coordinates.

		Args:
			x_in (list): transpose of h
						 (n components with values between 0 and 2**p-1)

		Returns:
			h (int): integer distance along hilbert curve
		"""
		x = list(x_in)
		if len(x) != self.n:
			raise ValueError('x={} must have N={} dimensions'.format(x, self.n))

		if any(elx > self.max_x for elx in x):
			raise ValueError(
				'invalid coordinate input x={}.  one or more dimensions have a '
				'value greater than 2**p-1={}'.format(x, self.max_x))

		if any(elx < 0 for elx in x):
			raise ValueError(
				'invalid coordinate input x={}.  one or more dimensions have a '
				'value less than 0'.format(x))

		M = 1 << (self.p - 1)

		# Inverse undo excess work
		Q = M
		while Q > 1:
			P = Q - 1
			for i in range(self.n):
				if x[i] & Q:
					x[0] ^= P
				else:
					t = (x[0] ^ x[i]) & P
					x[0] ^= t
					x[i] ^= t
			Q >>= 1

		# Gray encode
		for i in range(1, self.n):
			x[i] ^= x[i-1]
		t = 0
		Q = M
		while Q > 1:
			if x[self.n-1] & Q:
				t ^= Q - 1
			Q >>= 1
		for i in range(self.n):
			x[i] ^= t

		h = self._transpose_to_hilbert_integer(x)
		return h

def get_pc_boundingbox(minPt, maxPt): 
	dis = np.asarray(maxPt) - np.asarray(minPt)
	max_dis = max(dis)/2.0
	center = (np.asarray(maxPt) + np.asarray(minPt))/2.0
	[x0, y0, z0] = np.asarray(center) - [max_dis, max_dis, max_dis]
	[x1, y1, z1] = np.asarray(center) + [max_dis, max_dis, max_dis]
	return [[x0, y0, z0], [x1, y1, z1], max_dis*2]

def hilbert_sfc_traversal(geo_arr):
	geo_arr = np.asarray(geo_arr)
	minPt = [min(geo_arr[:,0]), min(geo_arr[:,1]), min(geo_arr[:,2])]
	maxPt = [max(geo_arr[:,0]), max(geo_arr[:,1]), max(geo_arr[:,2])]
	[lb_pt, rt_pt, radius] = get_pc_boundingbox(minPt, maxPt)
	p=10
	N=3
	hilbert_curve = HilbertCurve(p, N)

	cell_num = 2**p
	radius = radius*1.00001
	cell_len = radius/cell_num
	dic = dict()
	
	i = 0
	for pt in geo_arr:
		x_pos = int((pt[0] - lb_pt[0])/cell_len)
		y_pos = int((pt[1] - lb_pt[1])/cell_len)
		z_pos = int((pt[2] - lb_pt[2])/cell_len)
		coords = [x_pos, y_pos, z_pos]
		dist = hilbert_curve.distance_from_coordinates(coords)
		if not dist in dic:
			dic[dist] = []
		dic[dist].append(i)
		i = i + 1

	hilbert_traversal_idx_arr = []
	for i in range(0, 2**(N*p)):
		if i in dic:
			for idx in dic[i]:
				hilbert_traversal_idx_arr.append(idx)

	return hilbert_traversal_idx_arr

def traversal_order_visualization(traversal_idx_arr, geo_arr):
	vis_geo = []
	vis_rgb = []

	colors = cm.viridis(np.linspace(0, 1, len(traversal_idx_arr))) # other color schemes: gist_rainbow, nipy_spectral, plasma, inferno, magma, cividis
	for t in range(len(traversal_idx_arr)):
		idx = traversal_idx_arr[t]
		rgb = colors[t][0:3]
		vis_geo.append(geo_arr[idx])
		vis_rgb.append(rgb)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(vis_geo)
	pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_rgb))
	o3d.visualization.draw_geometries([pcd])

# Bjontegaard_metric
def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
	#https://github.com/Anserw/Bjontegaard_metric
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
		# See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
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
	frame_id = pc_id_arr[3]

	ply_path = "ply/" + frame_id + ".ply"
	
	#Seed points
	sv_pt_num = 256 # Average point number of supervoxels, it can also be set to smaller one (e.g. 128 or 64)
	off_path = "LOD_off/" + frame_id + "_n"+ str(sv_pt_num) + ".off" #

	off_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(sv_pt_num) + ".off"
	ply_path = floder + frame_id + ".ply"

	[geo_arr, rgb_arr, y_comp_arr, point_num, pc_width] = init(ply_path)
	[off_geo_arr, off_rgb_arr, off_pt_num] = read_off(off_path, geo_arr, rgb_arr)

	# Supervoxel generation
	off_ply_assign_dic = assign_ply_to_off(off_geo_arr, geo_arr, vis_flag=False)

	## Octree depth first traversal
	# octree_dft_idx_arr = octree_depth_first_traversal(geo_arr)
	# octree_dft_traversal_color_arr = [rgb_arr[idx][::-1] for idx in octree_dft_idx_arr]

	## 3D Hilbert space-filling curve based traversal
	# hilbert_traversal_idx_arr = hilbert_sfc_traversal(geo_arr)
	# hilbert_traversal_color_arr = [rgb_arr[idx][::-1] for idx in hilbert_traversal_idx_arr]

	## Binary space partition (BSP) based universal traversal (with/without tsp)
	# bsp_traversal_idx_arr = bsp_traversal(geo_arr) # This function can also be used to traversal the whole point cloud
	# bsp_traversal_color_arr = [rgb_arr[idx][::-1] for idx in bsp_traversal_with_tsp_idx_arr]
	bsp_traversal_with_tsp_idx_arr = bsp_traversal_with_tsp(geo_arr, off_geo_arr, off_ply_assign_dic)
	# bsp_traversal_with_tsp_color_arr = [rgb_arr[idx][::-1] for idx in bsp_traversal_with_tsp_idx_arr]

	# sfc_based_attr_img = hybrid_space_filling(bsp_traversal_with_tsp_color_arr, mb_size = 16, hori_snake_b = 4)
	# [bpp_arr, psnr_arr] = img_compression(sfc_based_attr_img, point_num, mb_size = 16, cmp_method = "webp", extra_bit = 0) #cmp_method = "webp"
	# print([bpp_arr, psnr_arr])

	# traversal_order_visualization(bsp_traversal_with_tsp_idx_arr, geo_arr)

	isomap_based_attr_img_gen(bsp_traversal_with_tsp_idx_arr, geo_arr, rgb_arr, mb_size = 16, hori_snake_b = 4)

