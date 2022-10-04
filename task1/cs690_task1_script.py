# %% [markdown]
# ### Imports

# %%
import os
import csv
import re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import SpaGCN as spg
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
import cv2

# %%
spg.__version__

# %% [markdown]
# ### Read in data
# The current version of SpaGCN requres three input data: 
# <br>
# 1. The gene expression matrix(n by k): expression_matrix.h5;
# <br>
# 2. Spatial coordinateds of samplespositions.txt;
# <br>
# 3. Histology image(optional): histology.tif, can be tif or png or jepg.
# <br>
# The gene expreesion data can be stored as an AnnData object. AnnData stores a data matrix .X together with annotations of observations .obs, variables .var and unstructured annotations .uns. 

# %%
# Read original 10x_h5 data and save it to h5ad
from scanpy import read_10x_h5
adata = read_10x_h5("./data/Train_filtered_feature_bc_matrix.h5")
spatial=pd.read_csv("./data/Train_tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0) 
adata.obs["x1"]=spatial[1]
adata.obs["x2"]=spatial[2]
adata.obs["x3"]=spatial[3]
adata.obs["x4"]=spatial[4]
adata.obs["x5"]=spatial[5]
adata.obs["x_array"]=adata.obs["x2"]
adata.obs["y_array"]=adata.obs["x3"]
adata.obs["x_pixel"]=adata.obs["x4"]
adata.obs["y_pixel"]=adata.obs["x5"]
#Select captured samples
adata=adata[adata.obs["x1"]==1]
adata.var_names=[i.upper() for i in list(adata.var_names)]
adata.var["genename"]=adata.var.index.astype("str")
adata.write_h5ad("./data/Train_filtered_feature_bc_matrix.h5ad")

# %%
#Read in gene expression and spatial location
adata=sc.read("./data/Train_filtered_feature_bc_matrix.h5ad")

#Read in histology image

# img=cv2.imread("./data/Train_tissue_hires_image.png")
# print(img.shape)

img=cv2.imread("./data/histology.tif")
print(img.shape)

# %% [markdown]
# ### Integrate gene expression and histology into a Graph

# %%
#Set coordinates
x_array=adata.obs["x_array"].tolist()
y_array=adata.obs["y_array"].tolist()
x_pixel=adata.obs["x_pixel"].tolist()
y_pixel=adata.obs["y_pixel"].tolist()

#Test coordinates on the image
img_new=img.copy()
for i in range(len(x_pixel)):
    x=x_pixel[i]
    y=y_pixel[i]
    img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

cv2.imwrite('./data/graph.jpg', img_new)

# %% [markdown]
# - The "s" parameter determines the weight given to histology when calculating Euclidean distance between every two spots. ‘s = 1’ means that the histology pixel intensity value has the same scale variance as the (x,y) coordinates, whereas higher value of ‘s’ indicates higher scale variance, hence, higher weight to histology, when calculating the Euclidean distance. 
# 
# - The "b" parameter determines the area of each spot when extracting color intensity.

# %%
#Calculate adjacent matrix
s=1
b=7
adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
#If histlogy image is not available, SpaGCN can calculate the adjacent matrix using the fnction below
#adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
np.savetxt('./data/adj.csv', adj, delimiter=',')

# %% [markdown]
# ### Spatial domain detection using SpaGCN

# %% [markdown]
# #### Expression data preprocessing

# %%
adata=sc.read("./data/Train_filtered_feature_bc_matrix.h5ad")
adj=np.loadtxt('./data/adj.csv', delimiter=',')
adata.var_names_make_unique()
spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
spg.prefilter_specialgenes(adata)
#Normalize and take log for UMI
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

# %% [markdown]
# #### Set hyper-parameters

# %% [markdown]
# - p: Percentage of total expression contributed by neighborhoods
# - l: Parameter to control p

# %%
p=0.5 
#Find the l value given p
l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

# %% [markdown]
# - n_clusters: Number of spatial domains wanted.
# - res: Resolution in the initial Louvain's Clustering methods. If the number of clusters is known, we can use the spg.search_res() function to search for suitable resolution(optional)

# %% [markdown]
# > NOTE: Since louvain's implementation is causing some bug due to excessive RAM usage, we modified the source code of SpaGCN repository shifted to leidenalg clustering method.

# %%
#If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
#For this toy data, we set the number of clusters=7 since this tissue has 7 layers
n_clusters=7
#Set seed
r_seed=t_seed=n_seed=100
#Seaech for suitable resolution
res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

# %%
res

# %% [markdown]
# #### Run SpaGCN

# %%
clf=spg.SpaGCN()
clf.set_l(l)
#Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
#Run
clf.train(adata,adj,init_spa=True,init="leidenalg",res=res, tol=5e-3, lr=0.05, max_epochs=200,n_clusters=7)
y_pred, prob=clf.predict()
adata.obs["pred"]= y_pred
adata.obs["pred"]=adata.obs["pred"].astype('category')
#Do cluster refinement(optional)
#shape="hexagon" for Visium data, "square" for ST data.
adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
adata.obs["refined_pred"]=refined_pred
adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
#Save results
adata.write_h5ad("./results/results.h5ad")

# %% [markdown]
# #### Plot spatial domains

# %%
adata=sc.read("./results/results.h5ad")
#Set colors used
plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
#Plot spatial domains
domains="pred"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig("./results/pred.png", dpi=600)
plt.close()

#Plot refined spatial domains
domains="refined_pred"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig("./results/refined_pred.png", dpi=600)
plt.close()

# %% [markdown]
# #### Output csv

# %%
adata.obs["pred"].to_csv("./results/pred_output.csv", header=False)
adata.obs["refined_pred"].to_csv("./results/refined_pred_output.csv", header=False)

# %% [markdown]
# ## References
# 
# - https://github.com/jianhuupenn/SpaGCN
# - https://www.nature.com/articles/s41592-021-01255-8


