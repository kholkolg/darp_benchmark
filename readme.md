demand generation
1. Create parameters in config.ini like:
> [my_city_30]\
> city = MyCity\
> name = mycity_300_mindist250\
> num_requests = 30000\
>...\
> save_shp = False

2. run from command line:
 > pytnon3 demand_generation my_city_30
 
Parameters' meaning:
* **city** - str, name of the city; used to download map by osmnx.
* **name** - str, name of instance; added to output directory
* **inputDir** - str, directory with input files (nodes.csv, edges.csv).
* **outputDir** - str, output directory for results.
* **numRequests** - int, number of trips in the instance
* **peaks** - peak hours separated by ';'. End and start of the peak are 
separated by comma. Ex: 6, 8; 15, 17
* **minDistance** - float, minimum Euclidean distance between the origin
and the destination
* **avgDistance** - float, average distance
* **clusterSize** - float, size of cluster, km2
* **crsGeo** - int, initial crs
* **crsMetric** - int, crs for metric projection
* **unusedRoads** - str, forbidden osm tags for nodes used in demand generation.
(nodes are kept in the graph.)  
* **saveShapefiles** - bool, save shapefiles

(**Don't use quotation marks for strings in .ini.** \
All values are read as 
strings and parsed inside the script.)



 