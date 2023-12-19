IN_DIRECTORY=$1

for idir in ${IN_DIRECTORY}*; do
    RAY_TUNING_DIR="${idir}/ray_tuning/";
    TENSORBOARD_DIR="${idir}/tensorboard/";
	
    rm ${RAY_TUNING_DIR}*;
    rm -r ${TENSORBOARD_DIR}*;	
done
