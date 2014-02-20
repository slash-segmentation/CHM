#!/bin/sh

declare -r FULLRUN_MODE="fullrun"

function usage() {

  echo -e "Run CHM via Panfish.

This program packages and runs CHM on grid compute resources via Panfish.

runCHMJobViaPanfish.sh <mode> <optional arguments>
  mode         Mode to run.  This script has several modes
               $FULLRUN_MODE   -- Runs CHM on all images using Panfish.

Optional Arguments:
  -h           Displays this help.
  -n           Specifies job name to pass to give to SGE.  No
               funny characters other then _ and only a-z|A-Z
               for the first character
                
"
  exit 1
}

#
# function called when USR2 signal is caught
#
on_usr2() {
  echo "USR2 signal caught exiting.." > $OUTPUT_DIR/KILL.JOB.REQUEST
  checkForKillFile $OUTPUT_DIR
  jobFailed "USR2 signal caught exiting.."
}

# trap usr2 signal cause its what gets sent by SGE when qdel is called
trap 'on_usr2' USR2

#    
# Submit job(s) via Panfish
# 
# castJob($iteration,$task arg)
#
# $iteration is the current try count ie 1,2,3
#
# $task arg can be one of two things, a file or a string in the
# format of #-#
# if its a file ie $OUTPUT_DIR/failed.jobs then --taskfile <VALUE>
# is passed to panfishcast.  If its #-# then that value is passed via -t <VALUE>
# to panfishcast 
# castJob(iteration,jobDir,taskArgument,jobName)
#
function castJob {

  local iteration=$1
  local jobDir=$2
  local taskArgument=$3
  local jobName=$4

  logStartTime "CastJob Iteration $iteration"
  local castStartTime=$START_TIME 
  local curdir=`pwd`
  cd $jobDir

  local taskArg="-t $taskArgument"
  # Depending on the argument use either the --taskfile or -t flag
  if [ -e "$t_arg" ] ; then
    taskArg="--taskfile $taskArgument"
  fi

  # Move the cast.out file out of the way
  if [ -e "$jobDir/$CAST_OUT_FILE" ] ; then
     /bin/mv $jobDir/$CAST_OUT_FILE $jobDir/cast.$(( $iteration - 1)).out
  fi

  $CASTBINARY $taskArg -q $CHUMMEDLIST -N $jobName $BATCH_AND_WALLTIME_ARGS --writeoutputlocal -o $jobDir/out/log/jobout/\$JOB_ID\.\$TASK_ID.stdout -e $jobDir/out/log/joberr/\$JOB_ID\.\$TASK_ID.stderr $jobDir/runCHM.sh $MATLAB_DIR > $jobDir/$CAST_OUT_FILE

  if [ $? != 0 ] ; then
      logEndTime "CastJob Iteration $iteration" $castStartTime 1
      jobFailed "Error running $CASTBINARY"
      cd $curdir
  fi

  # output in cast.out file will look like this
  # Your job-array 142.1-1:1 ("line") has been submitted
  # Your job-array 143.3-3:1 ("line") has been submitted
  # Your job-array 144.5-11:1 ("line") has been submitted

  cd $curdir
  logEndTime "CastJob Iteration $iteration" $castStartTime 0
}

#     
# Wait for jobs to complete
#
function waitForJobs {
  local iteration=$1
  local jobDir=$2

  logStartTime "WaitForJobs Iteration $iteration" 
  local waitForJobsStartTime=$START_TIME
  local sleepTime=60
  local jobStatus="NA"

  DOWNLOAD_DATA_REQUEST="DOWNLOAD.DATA.REQUEST"

  while [ "$jobStatus" != "done" ] 
  do

      checkForKillFile "$jobDir"

      logMessage "Iteration $iteration job status is $jobStatus.  Sleeping $sleepTime seconds"
      sleep $sleepTime

      if [ -e "$jobDir/$DOWNLOAD_DATA_REQUEST" ] ; then
          /bin/rm -f "$jobDir/$DOWNLOAD_DATA_REQUEST"
          logMessage "$DOWNLOAD_DATA_REQUEST file found.  Performing download"
          landData $iteration $CHUMMEDLIST "$jobDir" " --exclude CHM.tar.gz --exclude runCHM.sh.config --exclude *.sh --exclude *.out --exclude job.properties --exclude *.config"
          /bin/rm -f "$jobDir/$DOWNLOAD_DATA_REQUEST"
          logMessage "Removing $DOWNLOAD_DATA_REQUEST file"
      fi

      # Get status of job which will be set in JOBSTATUS 
      getStatusOfJobsInCastOutFile "$jobDir"
  done
  logEndTime "WaitForJobs Iteration $iteration" $waitForJobsStartTime 0
}

#
# Verify Results
# verifyResults(iteration,jobDir,jobs,writeToFailFile)
#
function verifyResults {
  local iteration=$1
  local jobDir=$2
  local jobs=$3
  local writeToFailFile=$4

  logStartTime "VerifyResults $iteration"
  local verifyResultsStartTime=$START_TIME
 
  local numFailedJobs=0
  
  local anyFailed="NO"

  # if the $jobs parameter is a file then this is the test
  # run and we should only verify the jobs that are in the
  # file passed in have successfully completed.  Otherwise
  # assume $jobs is of #-# format and we should check all jobs
  # in that range.
  if [ -e "$jobs" ] ; then
    TASKS_IN_FAIL_FILE=`wc -l $jobs`
    logMessage "Found $jobs file.  Only examining $TASKS_IN_FAIL_FILE tasks"
    for Y in `cat $jobs` ; do
      checkSingleTask "$jobDir" $Y
      if [ $? != 0 ] ; then
        anyFailed="YES"
        let numFailedJobs++
        if [ "$WRITE_TO_FAIL_FILE" == "yes" ] ; then
          echo "$Y" >> "$jobDir/failed.jobs.tmp"
        fi
      fi
    done
  else 
    local seqLine=`echo $jobs | sed "s/-/ /"`
    local lastOne=`echo $jobs | sed "s/^.*-//"`

    logMessage "Examining up to $lastOne tasks"

    for Y in `seq $seqLine` ; do
      checkSingleTask "$jobDir" $Y
      if [ $? != 0 ] ; then
        anyFailed="YES"
        let numFailedJobs++
        if [ "$writeToFailFile" == "yes" ] ; then
          echo "$Y" >> "$jobDir/failed.jobs.tmp"
        fi
      fi
    done
  fi
  
  local verifyExit=0

  if [ "$writeToFailFile" == "yes" ] ; then
    if [ -e "$jobDir/failed.jobs" ] ; then
       /bin/mv "$jobDir/failed.jobs" $jobDir/failed.$(( $iteration - 1 )).jobs
    fi
  fi

  # make sure we have a sorted unique list of failed jobs
  if [ "$anyFailed" == "YES" ] ; then
     verifyExit=1
     if [ "$writeToFailFile" == "yes" ] ; then
       logMessage "Creating failed.jobs file"
       cat "$jobDir/failed.jobs.tmp" | sort -g | uniq > "$jobDir/failed.jobs"
       /bin/rm -f "$jobDir/failed.jobs.tmp"
     fi
  fi

  NUM_FAILED_JOBS=$numFailedJobs

  logEndTime "VerifyResults $iteration" $verifyResultsStartTime $verifyExit
  return $verifyExit
}

#
# Checks that a single task ran successfully.  
# A task is considered successful if in folder there is an output
# image and a log file exists with no errors in the file.
# if there are multiple log files then the youngest is parsed
# checkSingleTask(jobDir,jobToCheck)
function checkSingleTask {
   local jobDir=$1
   local jobToCheck=$2

   # Verify we got a resulting file and it has a size greater then 0.
   local outputImageName=`egrep "^${jobToCheck}:::" "$jobDir/$RUN_CHM_CONFIG" | sed "s/^.*::://" | head -n 1 | sed "s/^.*\///"`
   if [ ! -s "$jobDir/out/${outputImageName}" ] ; then
         logWarning "No output image found for task $jobToCheck"
         return 1
   fi

   getSingleCHMTaskLogFile "$jobDir" "$jobToCheck"

   return $?
}

#
# run jobs up to value specified with multiple retries built in
#
# runJobs(jobDir,jobs,jobnName)
#
function runJobs {

  local jobDir=$1
  local jobs=$2
  local jobName=$3

  logStartTime "RunJobs"
  local runJobsStartTime==$START_TIME
 
  local runJobsExit=1

  local jobsToRun=$jobs

  if [ ! -e "$jobDir/$RUN_CHM_CONFIG" ] ; then
     jobFailed "No $jobDir/$RUN_CHM_CONFIG found. Did you run $CREATE_PRETRAIN_MODE ?"
  fi
  local firstImage=`egrep "1:::" $jobDir/$RUN_CHM_CONFIG | head -n 1 | sed "s/^.*::://"`
  local inputData=`dirname $firstImage`

  # set CHUMMEDLIST to list of clusters
  getInitialClusterList "$jobDir"

  if [ $? != 0 ] ; then
     jobFailed "Unable to get suitable list of clusters"
  fi

  local iteration=0

  while [ $iteration -le $MAX_RETRIES ] 
  do
     # Move old panfish job folders out of way
     moveOldClusterFolders $(( $iteration - 1 )) "$jobDir"

     # if there is failed jobs file use that
     if [ -e "$jobDir/failed.jobs" ] ; then
        logMessage "Found failed.jobs file using that to determine what jobs to run"
        jobsToRun="$jobDir/failed.jobs"
     fi

     if [ $iteration -gt 1 ] ; then
       checkForKillFile "$jobDir"
       logMessage "Iteration $iteration.  Jobs failed in previous iteration. sleeping $RETRY_SLEEP seconds before trying again"
       sleep $RETRY_SLEEP
     fi

     checkForKillFile "$jobDir"

     # Upload input image data
     chumData $iteration $CHUMMEDLIST $inputData "$jobDir/$CHUM_OUT_FILE"
     if [ $? != 0 ] ; then
        jobFailed "Unable to upload input image directory"
     fi
     
     checkForKillFile "$jobDir"

     # Upload job directory
     chumData $iteration $CHUMMEDLIST "$jobDir" "$jobDir/$CHUM_OUT_FILE" "--deletebefore --exclude *.tif --exclude *.tiff --exclude *.png --exclude *.out --exclude *.log --exclude *.jobs --exclude *.list --exclude *.old --exclude *.stderr --exclude *.stdout"
     if [ $? != 0 ] ; then
        jobFailed "Unable to upload job directory"
     fi

     checkForKillFile "$jobDir"

     # Submit job via Panfish
     castJob $iteration "$jobDir" "$JOBS_TO_RUN" "$jobName"
     
     # Wait for jobs to complete
     waitForJobs $iteration "$jobDir"

     # Download completed job results
     landData $iteration $CHUMMEDLIST "$jobDir" " --exclude CHM.tar.gz --exclude runCHM.sh.config --exclude *.sh --exclude *.out --exclude job.properties --exclude *.config"
     
     if [ $? != 0 ] ; then
        logMessage "Sleeping 60 seconds and retrying download..."
        sleep 60
        landData $iteration $CHUMMEDLIST "$jobDir" " --exclude CHM.tar.gz --exclude runCHM.sh.config --exclude *.sh --exclude *.out --exclude job.properties --exclude *.config"
        if [ $? != 0 ] ; then
           logWarning "Download failed a second time"
        fi
     fi
     
     checkForKillFile "$jobDir"

     # Verify Results
     verifyResults $iteration "$jobDir" "$jobs" "yes"
  
     # If nothing failed we are good.  Get out of the loop
     if [ "$ANYFAILED" == "NO" ] ; then
        runJobsExit=0
        break
     fi

     # Increment the counter
     iteration=$(( $iteration + 1 ))
  done

  if [ $runJobsExit != 0 ] ; then
    logWarning "Error running jobs...."    
  fi

  logEndTime "RunJobs" $runJobsStartTime $runJobsExit

  return $runJobsExit
}

###########################################################
#
# Start of program
#
###########################################################

declare CHM_JOB_NAME="chm_job"
# get the directory where the script resides
declare SCRIPT_DIR=`dirname $0`
declare SCRIPTS_SUBDIR="$SCRIPT_DIR/scripts"


if [ $# -lt 1 ] ; then
  usage;
fi

declare -l MODE=$1

shift 1

while getopts ":h:n" o; do
  case "${o}" in
    h)
      usage
      ;;
    n)
      CHM_JOB_NAME="${OPTARG}"
      ;;
    *)
      jobFailed "Invalid argument: ${o}."
      ;;
    esac
done

# load the helper functions
if [ -s "$SCRIPT_DIR/helperfuncs.sh" ] ; then
  . $SCRIPT_DIR/helperfuncs.sh
else 
  . $SCRIPTS_SUBDIR/helperfuncs.sh
fi

getFullPath "$SCRIPT_DIR"
declare OUTPUT_DIR="$GETFULLPATHRET"


# Parse the configuration file
parseProperties "$SCRIPT_DIR" "$OUTPUT_DIR"

if [ $? != 0 ] ; then
  jobFailed "There was a problem parsing the properties"
fi

logEcho ""
logStartTime "$MODE mode"
declare -i modeStartTime=$START_TIME
logEcho ""

#
# Full run mode
#
if [ "$MODE" == "$FULLRUN_MODE" ] ; then

  getNumberOfCHMJobsFromConfig $OUTPUT_DIR
  if [ $? != 0 ] ; then
     jobFailed "Error obtaining number of jobs"
  fi

  if [ ! -d "$MATLAB_DIR" ] ; then
    jobFailed "Unable to get path to matlab directory: $MATLAB_DIR"
  fi
 

  verifyResults 1 "$OUTPUT_DIR" "1-${NUMBER_JOBS}" "yes"
  declare verifyExit=$?

  logEcho ""

  if [ $NUM_FAILED_JOBS -eq 0 ] ; then
    logMessage "All $NUMBER_JOBS task(s) completed successfully"
    logEcho ""
    logEndTime "$MODE mode" $modeStartTime 0
    exit 0
  fi
    
  logMessage "$NUM_FAILED_JOBS out of $NUMBER_JOBS task(s) not completed."
  logMessage "Running jobs..."
  
  runJobs "$OUTPUT_DIR" "${THE_START_JOB}-${NUMBER_JOBS}" "$CHM_JOB_NAME"
  local runJobsExit=$?
  logEcho ""
  logEndTime "$MODE mode" $modeStartTime $runJobsExit

  exit $runJobsExit
fi

# implies an unsupported mode
jobFailed "Mode $MODE not supported.  Invoke with -h for list of valid options."

