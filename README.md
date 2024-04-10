**Considerations with this reduced data implementation**

*-There are two vehicle data sets provided for demonstration purposes. The full data set was not provided due to the file size considerations. The analysis referenced in the paper was completed with 200 vehicles of data. With additional full data files in <vehicle_data/> and <random_list/> folders, the program will run as full implementation.*

*-Folders will be created when training with models and data including: <data/>, <models_train/>, <model_train_ft_combined/>, <data/batch_gen_cf_oos140-200_true/>, and <vae_data/>*

*-Implementation assumes SLURM and CUDA.* 

*-See requirements.txt for software installation requirements.*

**Step 1: Run in parallel.**

->Train DeepAR Model with model 08_Train_Deep_AR_Fault_Spec_CUDA_slurm.py in working directory: command <sbatch -a 0-139 run_08.sh>. This will create <model_train_ft_combined/> folder with models.

->Train STAM over randomly selected time periods with file 06ga_ATTN_Training_Feature_Gen_CUDA.py: command <python 06ga_ATTN_Training_Feature_Gen_CUDA.py>. This will create <data/> and <models_train/> folders with models and data.

->Generate data for historical out of sample vehicle data using 06gv_Gen_CVAE_data_CUDA.py: command <python 06gv_Gen_CVAE_data_CUDA.py>. This will create <vae_data/> folder with data.

**Step 2: Run in parallel**

->Generated STAM data with file 07ga_ATTN_Generate_data_CUDA.py: command <python 07ga_ATTN_Generate_data_CUDA.py>. This will update in <data/> folder with data.

->Train VAE model with file 07gvA_VAE_Gen_CUDA.py: command <python 07gvA_VAE_Gen_CUDA.py>. This will update in <vae_data/> folder with VAE model.

**Step 4: Run in parallel**

->Train Kmeans model with file 07gvB_kmeans_CUDA.py: command <python 07gvB_kmeans_CUDA.py>.  This will update in <vae_data/> folder with Kmeans model and plots.

->Generate final STAM hidden data with file 09ga_ATTN_Eval_slurm.py: command <sbatch -a 0-139 run_09ga.sh> with file 09ga_ATTN_Eval_slurm.py in working directory. This will update in <data/> folder with data.

**Step 5: Run in parallel**
->Sample data and generate final VAE hidden data with file evaluating the DeepAR Model with 09gv_ATTN_Eval_slurm.py: <sbatch -a 0-139 run_09gv.sh> with model 09gv_ATTN_Eval_slurm.py in working directory. This will update in <data> folder with data.

**Step 6**
->Complete classificaiton analysis in python notebook: <classification_notebook.ipynb> 







