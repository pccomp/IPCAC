import open3d as o3d
import numpy as np
from random import randrange
from sklearn.neighbors import KNeighborsClassifier
from tsp_solver.greedy import solve_tsp

#################################################################### (1) Initialization 
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

#################################################################### (2) 3D to 1D
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
			sub_geo_idx_arr = partition_arr_idx[t]
			seg_geo_arr = [geo_arr[id] for id in sub_geo_idx_arr]
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
						temp_pt_idx_arr1.append(sub_seg_geo_idx_arr[i])
					else:
						temp_pt_idx_arr2.append(sub_seg_geo_idx_arr[i])

				if len(temp_pt_idx_arr1):
					temp_partition_arr.append(temp_pt_idx_arr1)

				if len(temp_pt_idx_arr2):
					temp_partition_arr.append(temp_pt_idx_arr2)
			else:
				temp_partition_arr.append(sub_seg_geo_idx_arr)

		partition_arr_idx = temp_partition_arr

		flag = 0
		for sub_seg_geo_idx_arr in partition_arr_idx:
			if len(sub_seg_geo_idx_arr)>1:
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

	st_idx = 0
	end_idx = 0
	pre_last_pt = []
	tsp_content_color_huff2 = []
	for t in range(len(off_bsp_traversal_idx)):
		idx = off_bsp_traversal_idx[t]
		nex_off_pt = off_geo_arr[idx]
		curr_cluster_geo = [geo_arr[id] for id in off_pt_assign_dic[idx]]
		curr_cluster_idx = off_pt_assign_dic[idx]
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
			dist_m = construct_tsp_distance_matrix(curr_cluster_geo, nei = 8)
			cluster_path = solve_tsp(dist_m, endpoints=(st_idx, end_idx))
		else:
			if len(curr_cluster_geo):
				cluster_path = [0]

		for idx in cluster_path:
			bsp_traversal_with_tsp_idx_arr.append(curr_cluster_idx[idx])

	return bsp_traversal_with_tsp_idx_arr



if __name__ == '__main__':
	pc_id_arr = ["andrew9_frame0027", "David_frame0000", "ricardo9_frame0039", "phil9_frame0244", "sarah9_frame0018", "Staue_Klimt", "Egyptian_mask", "Shiva_00035", "Facade_00009", "House_without_roof_00057", "Frog_00067", "Arco_Valentino_Dense"]
	frame_id = pc_id_arr[0]

	ply_path = "ply/" + frame_id + ".ply"
	
	#Seed points
	sv_pt_num = 256 # Average point number of supervoxels
	off_path = "LOD_off/" + frame_id + "_n"+ str(sv_pt_num) + ".off" #

	off_path = floder + "LOD_off" + "/" + frame_id + "_n"+ str(sv_pt_num) + ".off"
	ply_path = floder + frame_id + ".ply"

	[geo_arr, rgb_arr, y_comp_arr, point_num, pc_width] = init(ply_path)
	[off_geo_arr, off_rgb_arr, off_pt_num] = read_off(off_path, geo_arr, rgb_arr)

	# Supervoxel generation
	off_ply_assign_dic = assign_ply_to_off(off_geo_arr, geo_arr, vis_flag=False)

	#Binary space partition (BSP) based universal traversal (with/without tsp)
	# bsp_traversal_idx_arr = bsp_traversal(geo_arr) 
	bsp_traversal_with_tsp_idx_arr = bsp_traversal_with_tsp(geo_arr, rgb_arr, off_geo_arr, off_ply_assign_dic)
  
  #to be continued
