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
![Graph: comparison of LSHForest by n-estimators with threshold=2](images/lshf_accuracy_by_n-estimators_threshold=2.png)
That graph took hours to compute, mostly because of the n-estimators=15: the median time was 40 minutes for a single run-through of the model, while with n-estimators=10 the median time was 10 minutes. This seems to be the curse of dimensionality rearing its ugly head, so I'll keep n-estimators at or below 10. Surprisingly, though, n-candidates was only modestly correlated with time taken by each model, so it seems we can have a higher value for that hyperparameter.
```
  n_candidates  n_estimators percent_correct time(minutes)
  1             5             0.2011          5.99
  11            5             0.3882          6.7
  21            5             0.4515          7.52
  31            5             0.4884          7.72
  41            5             0.5054          7.76
  51            5             0.5311          7.92
  1             10            0.2985          12.43
  11            10            0.5060          9.8
  21            10            0.5669          9.63
  31            10            0.6008          9.9
  41            10            0.6074          11.09
  51            10            0.6546          12.82
  1             15            0.3490          60.29
  11            15            0.5640          79.77
  21            15            0.6148          40.49
  31            15            0.6688          40.97
  41            15            0.6867          7.91
  51            15            0.6987          24.56
```
These aren't very good results. They all pale in comparison to nearest centroid, which gets better accuracy than all these with a fraction of the time.

### Separating correct and incorrect predictions by distance
![Histogram: comparison of distance between a vector and its nearest neighbor using LSHForest, separated by correct and incorrect predictions](images/lshf_distancediff_thresh=1_n-candidates=100.png)
