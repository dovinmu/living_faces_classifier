# Living Faces of the Wild: an exploration of models for facial recognition


### Thresholds
We can vary how many vectors we load into the dataset as a kind of hyperparameter for all our models. Basically we can exclude people with fewer than `threshold` vectors to see how much our accuracy goes up. For any neighbors-based supervised model, letting in all the data with a threshold of 1 guarantees that all the people with only one vector will be wrong since their nearest neighbors will always be another person.

<table>
	<tr>		<td>threshold</td>		<td>vectors</td>		<td>people</td>	</tr>
	<tr>		<td>1</td>		<td>13199</td>		<td>5736</td>	</tr>
	<tr>		<td>2</td>		<td>9137</td>		<td>1674</td>	</tr>
	<tr>		<td>4</td>		<td>6712</td>		<td>607</td>	</tr>
	<tr>		<td>6</td>		<td>5413</td>		<td>310</td>	</tr>
	<tr>		<td>8</td>		<td>4817</td>		<td>217</td>	</tr>
	<tr>		<td>10</td>		<td>4319</td>		<td>158</td>	</tr>
	<tr>		<td>20</td>		<td>3020</td>		<td>62</td>	</tr>
	<tr>		<td>30</td>		<td>2368</td>		<td>34</td>	</tr>
	<tr>		<td>40</td>		<td>1865</td>		<td>19</td>	</tr>
	<tr>		<td>50</td>		<td>1558</td>		<td>12</td>	</tr>
</table>

### Nearest centroid

As a baseline, we can start with Nearest Centroid. There are no hyperparameters for Nearest Centroid, so this is just a graph showing how well it performs as we exclude more and more of the sparser people we're trying to identify.

![Graph: Nearest centroid accuracy by threshold](images/nearest_centroid1.png)

That graph took 2 minutes to compute, with the following accuracy scores:
<table>
<tr><td>threshold</td><td>accuracy</td></tr>
<tr><td>1</td><td>0.550191</td></tr>
<tr><td>2</td><td>0.777904</td></tr>
<tr><td>4</td><td>0.931603</td></tr>
<tr><td>6</td><td>0.970807</td></tr>
<tr><td>8</td><td>0.970946</td></tr>
<tr><td>10</td><td>0.977169</td></tr>
<tr><td>20</td><td>0.987254</td></tr>
<tr><td>30</td><td>0.991678</td></tr>
<tr><td>40</td><td>0.987250</td></tr>
<tr><td>50</td><td>0.986577</td></tr>
</table>

### k-nearest neighbors
![Graph: comparison of k-nearest neighbors by n_neighbors and threshold](images/knn_2-50.png)
### Locality-sensitive hashing forest
![Graph: comparison of LSHForest by threshold](images/lshforest_n-candidates=100.png)
#### Separating correct and incorrect predictions by distance
![Histogram: comparison of distance between a vector and its nearest neighbor using LSHForest, separated by correct and incorrect predictions](images/lshf_distancediff_thresh=1_n-candidates=100.png)
