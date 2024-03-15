while read requirement
    do pip install $requirement || conda install --yes $requirement 
done < requirements.txt
conda install -c conda-forge pytorch-gpu
pip install --upgrade timm