d=$(date +%Y-%m-%d)

zip -r grid_results_$d.zip grid_results/ 

mv grid_results_$d.zip check_points/

zip -r learning_curves_$d.zip learning_curves/ 

mv learning_curves_$d.zip check_points/

zip -r folds_$d.zip folds/ 

mv folds_$d.zip check_points/