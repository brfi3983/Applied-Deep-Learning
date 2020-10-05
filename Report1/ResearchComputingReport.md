# Getting Started:
All the info is [here.](https://curc.readthedocs.io/en/latest/index.html)
First apply for an account at RC Computing at their [website](https://rcamp.rc.colorado.edu/accounts/account-request/create/organization).
 
* Set up double factor authentication with DUO
* Download DUO Mobile and have them connect it to your account
* Make sure you have Summit access in addition to PetaLibrary (allocations) so you can submit jobs

# Logging in:
### To login, type the following command in Command Prompt or Powershell:
```ssh $USER@login.rc.colorado.edu```

It will ask you for a password and then send a DUO 2FA to your phone through DUO Mobile.
### Once inside, you have two main folders: `</home/$USER>` and `</projects/$USER>`.
The `</home/>` folder is 2GB and the `</projects/>`folder is 250 GB. It is recommended you store scripts on `</home/>` and data or packages on `</projects/>`

Check space on these folders with ```curc-quota```; this will give you different answers depending on which node you are in. 

Each directory is also backed up into the `</.snapshot/>` hidden directory.

# Nodes:
You have three different nodes for various reasons (mainly computational reasons). I will just paste what they have on their site.

### Node types
Research Computing has several node types available on our resources. Each node type is meant for certain tasks. These node types are relatively common for other HPC centers. We will discuss each node type and its intended use below.

### Login nodes
* Four virtual machines
* This is where you are when you log in
* No computation, compiling code, interactive jobs, or long running processes
* Script or code editing
* Job submission

### Compile nodes
* Where you compile code, such as Fortran, C, C++
* No heavy computation
* Job submission
* Access these nodes by typing ssh scompile from a login node

### Compute nodes
* This is where jobs that are submitted through the scheduler run.
* Intended for heavy computation
* When run an interactive job will be performing tasks directly on the compute nodes

# Submitting Jobs:
To submit a batch job (a job that does not run interactively), we create a shell script and call the program within the script
There are various parameters you can setup to take advantage of different resources - more [here](https://curc.readthedocs.io/en/latest/running-jobs/batch-jobs.html).
A sample script:

	#!/bin/bash
	#SBATCH --partition=shas-testing %different partitions have different resources
	#SBATCH --nodes=1
	#SBATCH --ntasks=1
	#SBATCH --time=00:01:00 %max time you expect the job to take
	#SBATCH --job-name=test-job
	#SBATCH --output=test-job.%j.out
	
	module purge %clears all loaded modules
	source /curc/sw/anaconda3/latest %activates conda
	conda activate deeplearning %deeplearning is my enviroment
	
	python torch_testing.py

I would urge you to look at the different paramters as I do not know much about them. This is what worked for me in the end, so I stuck with it.
*You will notice that I have a conda enviroment, this will be important so we can load tensorflow, keras, pytorch, numpy, etc.*

Anyways, this script will be stored in a file called ```test-job.sh```. To submit it as a batch job, you type ```sbatch test-job.sh``` This will give you an ID number which you can check that status of. You can also see the output of this job (the output of your python program since it was called) by going to the file ```test-job.[JOBID].out```.

# Setting up your Conda Enviroment:
To setup conda, you first need to change the directory conda uses so you do not fill up your `</home/>` directory with packages
To do this, simply edit this file by typing ```nano ~/.condarc```
Then, paste the following four lines:

	pkgs_dirs:
		- /projects/$USER/.conda_pkgs
	envs_dirs:
		- /projects/$USER/software/anaconda/envs

This will stay as is unless you change it.
Now, activate the anaconda enviroment:

```source /curc/sw/anaconda3/latest```

Now you can use Conda!
I recommended you setup your own conda enviroment to store all packages on. If one wants to use python 3.6.8, you can type 

```conda create -n mycustomenv python==3.6.8``` 

Here, ```mycustomenv=deeplearning```. 
Now, you can install packages (such as numpy, scipy, and tensorflow) by typing 

```conda install numpy scipy tensorflow```. 

***Note***, it is recommended you try to install all your packages in one go, but you can add some later.

**This is why in the script we included those last three lines before calling our program. It activates conda, goes to your enviroment with all your packages, and then runs your program!**

# Useful Bash Commands:
	# To move files from and to the research enviroment and your local machine, use this:
	# # Local -> RC
	scp < path-to-file > <username > @login.rc.colorado.edu: < target-path >
	# # RC -> Local
	scp < username > @login.rc.colorado.edu: < path-to-file > <target-path >
	# To list components within directory:
	ls
	# To make a file:
	touch
	# To make a folder:
	mkdir
	# To change directories:
	cd /directory/
	# Going back to a previous directory:
	cd ..
	# Removing a file:
	rm "file"
	# Removing a folder:
	rm -rf "folder"
	# Moving a file within your directories:
	mv "PATH OF SOURCE" "PATH OF DESTINATION"
	# Seeing the contents of a file:
	cat file
	# Editing the contents of a file:
	nano file