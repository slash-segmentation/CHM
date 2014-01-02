#!/bin/sh

TESTRUN_MODE="-testrun"
PRINTONLY_MODE="-printonly"
FULLRUN_MODE="-fullrun"
CREATE_MODE="-create"
VERIFY_MODE="-verify"

if [ $# -lt 2 ] ; then
  echo "$0 <mode> <arguments>"
  echo ""
  echo "This program packages and runs CHM on grid compute resources via Panfish.  This program"
  echo "supports a number of modes as described below."
  echo ""
  echo "Supported modes:"
  echo ""
  echo "$CREATE_MODE <path to training data> <input data directory> <output directory> <stage> <level> <block size>"
  echo "     Creates job in <output directory> that is runnable via other commands below"
  echo "        <path to training data> -- Should be set to directory containing training data"
  echo "        <input data directory> -- Should be set to directory containing *.png files representing slices to process"
  echo "        <output directory> -- Directory where job will be created"
  echo "        <stage> -- Stage to use, most people set this to 2"
  echo "        <level> -- Level to use, most people set this to 2"
  echo "        <block size> -- Size of tiles to break slices into, most people use 1000 which results in 1k x 1k tiles"
  echo "                        NOTE:  Going above 1k block size may cause jobs to fail due to too much memory consumption"
  
  echo ""
  echo "$TESTRUN_MODE <output directory>"
  echo "     Runs CHM via Panfish on the first slice/image/png file"
  echo ""
  echo "$FULLRUN_MODE <output directory>"
  echo "     Runs CHM via Panfish on all slices"
  echo ""
  echo "$PRINTONLY_MODE <output directory>"
  echo "     Prints the commands this script would run but does NOT run them"
  echo ""
  echo "$VERIFY_MODE <output directory>"
  echo "     Checks jobs for successful completion"
  echo ""
  echo "Examples:"
  echo ""
  echo "$0 $CREATE_MODE /foo/Nstage2_Nlevel2 /foo/inputslices /share/chmrun 2 2 2000"
  echo ""
  echo "$0 $TESTRUN_MODE /share/chmrun"
  echo ""
  echo "$0 $FULLRUN_MODE /share/chmrun"
  echo ""
  echo "$0 $PRINTONLY_MODE /share/chmrun"
  echo ""
  exit 1
fi

# get the directory where the script resides
SCRIPT_DIR=`dirname $0`
SCRIPTS_SUBDIR="$SCRIPT_DIR/scripts"

# load the helper functions
. $SCRIPTS_SUBDIR/helperfuncs.sh

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
# Invoked in the print only mode.  This function
# outputs the panfish commands to run the
# job in its entirety.
#
function runPrintOnlyMode {
  echo ""
  echo "Print mode invoked.  Run the following commands:"
  echo ""

  CHUMMEDLIST=`$CASTBINARY --listclusters`

  # use Panfish chum to upload the OUTPUT_DIR
  echo "  # Use Panfish chum to upload $OUTPUT_DIR"
  echo "  panfishchum --path $OUTPUT_DIR --cluster $CHUMMEDLIST";
  echo ""

  getInputDataDirectory $OUTPUT_DIR

  # use Panfish chum to upload input data
  echo "  # Use Panfish chum to upload input data"
  echo "  panfishchum --path $INPUT_DATA --cluster $CHUMMEDLIST"
  echo ""

  getNumberOfCHMJobsFromConfig $OUTPUT_DIR

  # use Panfish cast to submit runCHM.sh array jobs 
  echo "  # Use Panfish cast to submit runCHM.sh array jobs"
  echo "  cd $OUTPUT_DIR ; panfishcast -t 1-${NUMBER_JOBS} -N chm_job -q $CHUMMEDLIST $OUTPUT_DIR/runCHM.sh"
  echo ""

  # wait for completion
  echo "  # Wait for completion of jobs use qstat or add -sync y to panfishcast command above"
  echo ""
  # download results
  echo "  # Download results"
  echo "  panfishland --path $OUTPUT_DIR --cluster $CHUMMEDLIST"
  echo ""
}


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
#
function castJob {
  # $1 iteration
  # $2 task argument
  logStartTime "CastJob Iteration $1"
 
  CURDUR=`pwd`

  cd $OUTPUT_DIR

  CAST_START_TIME=$START_TIME
  BATCH_AND_WALLTIME_ARGS="--batchfactor gordon_shadow.q::0.5,trestles_shadow.q::0.25,coleslaw_shadow.q::0.25 --walltime gordon_shadow.q::13:00:00,trestles_shadow.q::14:00:00"
  T_ARG=$2

  # Move the cast.out file out of the way
  if [ -e "$OUTPUT_DIR/$CAST_OUT_FILE" ] ; then
     /bin/mv $OUTPUT_DIR/$CAST_OUT_FILE $OUTPUT_DIR/cast.$(( $1 - 1)).out
  fi

  # Depending on the argument use either the --taskfile or -t flag
  if [ -e "$T_ARG" ] ; then
    TASK_ARG="--taskfile $T_ARG"
  else
    TASK_ARG="-t $T_ARG"
  fi

  logMessage "Running $CASTBINARY $TASK_ARG -q $CHUMMEDLIST -N chm_job $BATCH_AND_WALLTIME_ARGS $OUTPUT_DIR/runCHM.sh $MATLAB_DIR > $OUTPUT_DIR/$CAST_OUT_FILE"

  $CASTBINARY $TASK_ARG -q $CHUMMEDLIST -N chm_job $BATCH_AND_WALLTIME_ARGS $OUTPUT_DIR/runCHM.sh $MATLAB_DIR > $OUTPUT_DIR/$CAST_OUT_FILE

  if [ $? != 0 ] ; then
      logEndTime "CastJob Iteration $1" $CAST_START_TIME 1
      cd $CURDIR
      jobFailed "Error running $CASTBINARY$TASK_ARG -q $CHUMMEDLIST -N chm_job $BATCH_AND_WALLTIME_ARGS $OUTPUT_DIR/runCHM.sh $MATLAB_DIR > $OUTPUT_DIR/$CAST_OUT_FILE"
  fi

  # output in cast.out file will look like this
  # Your job-array 142.1-1:1 ("line") has been submitted
  # Your job-array 143.3-3:1 ("line") has been submitted
  # Your job-array 144.5-11:1 ("line") has been submitted

  cd $CURDIR
  logEndTime "CastJob Iteration $1" $CAST_START_TIME 0
}

#     
# Wait for jobs to complete
#
function waitForJobs {
  # $1 iteration
  logStartTime "WaitForJobs Iteration $1" 
  WAIT_FOR_JOBS_START_TIME=$START_TIME

  JOBSTATUS="NA"

  DOWNLOAD_DATA_REQUEST="DOWNLOAD.DATA.REQUEST"

  while [ "$JOBSTATUS" != "done" ] 
  do

      checkForKillFile $OUTPUT_DIR

      logMessage "Iteration $1 job status is $JOBSTATUS.  Sleeping 60 seconds"
      sleep 60


      if [ -e "$OUTPUT_DIR/$DOWNLOAD_DATA_REQUEST" ] ; then
          /bin/rm -f $OUTPUT_DIR/$DOWNLOAD_DATA_REQUEST
          logMessage "$DOWNLOAD_DATA_REQUEST file found.  Performing download"
          landData $1
          /bin/rm -f $OUTPUT_DIR/$DOWNLOAD_DATA_REQUEST
          logMessage "Removing $DOWNLOAD_DATA_REQUEST file"
      fi

      # Get status of job which will be set in JOBSTATUS 
      getStatusOfJobsInCastOutFile $OUTPUT_DIR
  done
  logEndTime "WaitForJobs Iteration $1" $WAIT_FOR_JOB_START_TIME 0
}

#
# Verify Results
#
function verifyResults {
  # $1 iteration
  # $2 last job
  # $3 optional print flag
  logStartTime "VerifyResults $1"
  VERIFY_RESULTS_START_TIME=$START_TIME
 
  let NUM_FAILED_JOBS=0
 
  WRITE_TO_FAIL_FILE="yes"
  
  if [ $# -eq 3 ] ; then
     WRITE_TO_FAIL_FILE=$3
  fi

  if [ "$WRITE_TO_FAIL_FILE" == "yes" ] ; then
    if [ -e $OUTPUT_DIR/failed.jobs ] ; then
       /bin/mv $OUTPUT_DIR/failed.jobs $OUTPUT_DIR/failed.$(( $1 - 1 )).jobs
    fi
  fi
  
  ANYFAILED="NO"

  SEQ_LINE=`echo $2 | sed "s/-/ /"`

  LAST_ONE=`echo $2 | sed "s/^.*-//"`

  logMessage "Examining $LAST_ONE tasks"

  for Y in `seq $SEQ_LINE` ; do
     checkSingleTask $Y
     if [ $? != 0 ] ; then
         ANYFAILED="YES"
         let NUM_FAILED_JOBS++
         if [ "$WRITE_TO_FAIL_FILE" == "yes" ] ; then
            echo "$Y" >> $OUTPUT_DIR/failed.jobs.tmp
         fi
     fi
  done

  logMessage "Examining logs from these clusters: $CHUMMEDLIST"

  for Y in `echo $CHUMMEDLIST | sed "s/,/ /g"` ; do
     if [ -d "$OUTPUT_DIR/$Y" ] ; then
        for Z in `grep "Exit Code:" $OUTPUT_DIR/${Y}/*stdout | grep -v "Exit Code: 0" | sed "s/^.*(task //" | sed "s/).*//" | sort | uniq` ; do
           ANYFAILED="YES"
           let NUM_FAILED_JOBS++
           if [ "$WRITE_TO_FAIL_FILE" == "yes" ] ; then
              logWarning "ERROR: A sub job in task $Z had non zero exit code.  Adding $Z to failed.jobs file"
              echo "$Z" | sed "s/^.*\.//" >> $OUTPUT_DIR/failed.jobs.tmp
           else 
              logWarning "ERROR: A sub job in task $Z had non zero exit code."
           fi
        done
     fi
  done

  VERIFY_EXIT=0

  # make sure we have a sorted unique list of failed jobs
  if [ "$ANYFAILED" == "YES" ] ; then
     VERIFY_EXIT=1
     if [ "$WRITE_TO_FAIL_FILE" == "yes" ] ; then
       cat $OUTPUT_DIR/failed.jobs.tmp | sort -g | uniq > $OUTPUT_DIR/failed.jobs
       /bin/rm -f $OUTPUT_DIR/failed.jobs.tmp
     fi
  fi

  logEndTime "VerifyResults $1" $VERIFY_RESULTS_START_TIME $VERIFY_EXIT
  return $VERIFY_EXIT
}

#
# Checks that a single task ran successfully.  
# A task is considered successful if in folder there is an output
# image and a log file exists with no errors in the file.
# if there are multiple log files then the youngest is parsed
#
function checkSingleTask {
   THEJOB=$1

   # Verify we got a resulting tif file and it has a size greater then 0.
   OUTPUT_IMAGE_NAME=$SCRATCH/`egrep "^${THEJOB}:::" $OUTPUT_DIR/$RUN_CHM_CONFIG | sed "s/^.*::://" | head -n 2 | tail -n 1 | sed "s/^.*\///"`
   if [ ! -s "$OUTPUT_DIR/out/${OUTPUT_IMAGE_NAME}" ] ; then
         logWarning "task # $THEJOB: $OUTPUT_DIR/out/${OUTPUT_IMAGE_NAME} is missing or zero size"
         return 1
   fi

   # Examine log file from runjob.sh and verify size is greater then 0
   getSingleCHMTaskLogFile $OUTPUT_DIR $THEJOB
   if [ $? != 0 ] ; then
        logWarning "task # $THEJOB: Log file missing"
        return 1
   fi

   # Examine log file from runjob.sh and verify there is no Caught Fatal Exception messages
   getFatalExceptionFromSingleCHMTask $OUTPUT_DIR $THEJOB
   CHECK_EXIT=$?
   if [ $CHECK_EXIT != 0 ] ; then
      logWarning "Error parsing log file for task # $THEJOB"
   elif [ "$FATAL_MESSAGE" != "" ] ; then
      logWarning "task # $THEJOB had fatal exception: $FATAL_MESSAGE"
      CHECK_EXIT=1
   fi

   return $CHECK_EXIT
}

#
# run jobs up to value specified with multiple retries built in
#
#
function runJobs {

  logStartTime "RunJobs"
  RUN_JOBS_START_TIME=$START_TIME
  JOBS=$1
  RUNJOBS_EXIT=1

  JOBS_TO_RUN=$JOBS

  if [ ! -e "$OUTPUT_DIR/$RUN_CHM_CONFIG" ] ; then
     jobFailed "No $OUTPUT_DIR/$RUN_CHM_CONFIG found. Did you run $CREATE_MODE ?"
  fi
  FIRST_IMAGE=`egrep "1:::" $OUTPUT_DIR/$RUN_CHM_CONFIG | head -n 1 | sed "s/^.*::://"`
  INPUT_DATA=`dirname $FIRST_IMAGE`

  # set CHUMMEDLIST to list of clusters
  getInitialClusterList $OUTPUT_DIR

  if [ $? != 0 ] ; then
     jobFailed "Unable to get suitable list of clusters"
  fi

  while [ $X -le $MAX_RETRIES ] 
  do
     # Move old panfish job folders out of way
     moveOldClusterFolders $(( $X - 1 )) $OUTPUT_DIR 

     # if there is failed jobs file use that
     if [ -e "$OUTPUT_DIR/failed.jobs" ] ; then
        logMessage "Found failed.jobs file using that to determine what jobs to run"
        JOBS_TO_RUN="$OUTPUT_DIR/failed.jobs"
     fi

     if [ $X -gt 1 ] ; then
       checkForKillFile $OUTPUT_DIR
       logMessage "Iteration $X.  Jobs failed in previous iteration. sleeping $RETRY_SLEEP seconds before trying again"
       sleep $RETRY_SLEEP
     fi

     checkForKillFile $OUTPUT_DIR

     # Upload input image data
     chumData $X $CHUMMEDLIST $INPUT_DATA $OUTPUT_DIR/$CHUM_OUT_FILE 
     if [ $? != 0 ] ; then
        jobFailed "Unable to upload input image directory"
     fi
     
     checkForKillFile $OUTPUT_DIR

     # Upload job directory
     chumData $X $CHUMMEDLIST $OUTPUT_DIR $OUTPUT_DIR/$CHUM_OUT_FILE "--exclude \"out/*.png\" --exclude \"*.out\""
     if [ $? != 0 ] ; then
        jobFailed "Unable to upload job directory"
     fi

     checkForKillFile $OUTPUT_DIR

     # Submit job via Panfish
     castJob $X $JOBS_TO_RUN
     
     # Wait for jobs to complete
     waitForJobs $X

     # Download completed job results
     landData $X $CHUMMEDLIST $OUTPUT_DIR " --exclude CHM.tar.gz --exclude runCHM.sh.config --exclude \"*.sh\" "
     
     if [ $? != 0 ] ; then
        logMessage "Sleeping 60 seconds and retrying download..."
        sleep 60
        landData $X $CHUMMEDLIST $OUTPUT_DIR " "
        if [ $? != 0 ] ; then
           logWarning "Download failed a second time"
        fi
     fi
     
     checkForKillFile $OUTPUT_DIR

     # Verify Results
     verifyResults $X $JOBS
  
     # If nothing failed we are good.  Get out of the loop
     if [ "$ANYFAILED" == "NO" ] ; then
        RUNJOBS_EXIT=0
        break
     fi

     # Increment the counter
     X=$(( $X + 1 ))
  done

  logEndTime "RunJobs" $RUN_JOBS_START_TIME $RUNJOBS_EXIT

  return $RUNJOBS_EXIT
}


function runCreateMode {

  # try to make the output directory
  makeDirectory $OUTPUT_DIR


  if [ ! -d $TRAINING_DATA ] ; then
    jobFailed "$TRAINING_DATA is supposed to be a directory"
  fi

  if [ ! -d $INPUT_DATA ] ; then
    jobFailed "$INPUT_DATA is supposed to be a directory"
  fi

  if [ ! -d $OUTPUT_DIR ] ; then
    jobFailed "$OUTPUT_DIR is supposed to be a directory"
  fi

 
  # Copy over CHM folder
  INPUT_CHM_FOLDER="$SCRIPT_DIR/CHM"
  
  logMessage "Copying over $INPUT_CHM_FOLDER to $OUTPUT_DIR"
  
  /bin/cp -a $INPUT_CHM_FOLDER $OUTPUT_DIR/.

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp -a $INPUT_CHM_FOLDER $OUTPUT_DIR/."
  fi


  # Create out directory under $OUTPUT_DIR/CHM folder
  makeDirectory $OUTPUT_DIR/CHM/out

  logMessage "Copying scripts to $OUTPUT_DIR"
  # Copy over runCHM.sh script
  /bin/cp $SCRIPTS_SUBDIR/runCHM.sh $OUTPUT_DIR/.

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp $SCRIPT_DIR/runCHM.sh $OUTPUT_DIR/."
  fi

  # Copy the runjob.sh into the CHM directory
  /bin/cp $SCRIPTS_SUBDIR/runjob.sh $OUTPUT_DIR/CHM/.

  if [ $? != 0 ] ; then
     jobFailed "Error running /bin/cp $SCRIPTS_SUBDIR/runjob.sh $OUTPUT_DIR/CHM/."
  fi

  # Copy the helperfuncs.sh
  /bin/cp $SCRIPTS_SUBDIR/helperfuncs.sh $OUTPUT_DIR/.

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp $SCRIPTS_SUBDIR/helperfuncs.sh $OUTPUT_DIR/."
  fi
  getSizeOfPath $TRAINING_DATA
  logMessage "Copying Model/Training Data $TRAINING_DATA which is $NUM_BYTES bytes in size to $OUTPUT_DIR/CHM/out/"
  # Copy training data into output folder under CHM folder
  /bin/cp -r $TRAINING_DATA/* $OUTPUT_DIR/CHM/out/.

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/cp -r $TRAINING_DATA/* $OUTPUT_DIR/CHM/out/."
  fi

  CURDIR=`pwd`
  getSizeOfPath $OUTPUT_DIR/CHM
  # Tar and gzip CHM into CHM.tar.gz
  cd $OUTPUT_DIR/
  logStartTime "Tarring and Gzipping CHM folder which is $NUM_BYTES bytes" 
  tar -cz CHM > CHM.tar.gz
  logEndTime "Tarring and Gzipping CHM folder which is $NUM_BYTES bytes" $START_TIME $?

  # Delete the CHM folder since we have the compressed copy now
  /bin/rm -rf CHM

  if [ $? != 0 ] ; then
    jobFailed "Error running /bin/rm -rf CHM"
  fi
 
  getSizeOfPath $OUTPUT_DIR/CHM.tar.gz
  logMessage "CHM.tar.gz compressed file is $NUM_BYTES in size"
 
  # switch back to the original directory
  cd $CURDIR

  # create out directory under $OUTPUT_DIR
  makeDirectory $OUTPUT_DIR/out/log



  logMessage "Generating $OUTPUT_DIR/runCHM.sh.config configuration file"
  # Examine INPUT_DATA and create a configuration file in 
  # OUTPUT_DIR with this format:
  # 1:::<full path to input png file>
  # 1:::<relative path to output tif file>
  # 1:::<nstage>
  # 1:::<nlevel>
  # 1:::<block size>
  # where 1 will be the SGE_TASK_ID for the job

  CONFIG_FILE="$OUTPUT_DIR/runCHM.sh.config"

  # Delete any existing configuration file
  /bin/rm -f $CONFIG_FILE

  let CNTR=0
  for Y in `find $INPUT_DATA -name "*.png" -type f | sort -n` ; do
    let CNTR++
    echo "$CNTR:::$Y" >> $CONFIG_FILE

    INPUT_FILE_NAME=`echo $Y | sed "s/\.png//" | sed "s/^.*\///"`
    echo "$CNTR:::CHM/out/combined_${INPUT_FILE_NAME}.tif" >> $CONFIG_FILE

    echo "$CNTR:::$STAGE" >> $CONFIG_FILE
    echo "$CNTR:::$LEVEL" >> $CONFIG_FILE
    echo "$CNTR:::$BLOCKSIZE" >> $CONFIG_FILE
  done
}

###########################################################
#
# Start of program
#
###########################################################

MODE=$1

if [ ! -s "$SCRIPT_DIR/panfishCHM.config" ] ; then
  jobFailed "$SCRIPT_DIR/panfishCHM.config file not found."
fi

# if set must end with /
PANFISH_BIN_DIR=`egrep "^panfish.bin.dir" $SCRIPT_DIR/panfishCHM.config | sed "s/^.*= *//"`

MATLAB_DIR=`egrep "^matlab.dir" $SCRIPT_DIR/panfishCHM.config | sed "s/^.*= *//"`

X=1
MAX_RETRIES=5
RETRY_SLEEP=100
CASTBINARY="${PANFISH_BIN_DIR}panfishcast"
CHUMBINARY="${PANFISH_BIN_DIR}panfishchum"
LANDBINARY="${PANFISH_BIN_DIR}panfishland"
PANFISHSTATBINARY="${PANFISH_BIN_DIR}panfishstat"
CHUMMEDLIST=""

logEcho ""
logStartTime "$MODE mode"
MODE_START_TIME=$START_TIME
logEcho ""

# 
# Create mode
#
if [ "$MODE" == "$CREATE_MODE" ] ; then

  if [ $# -ne 7 ] ; then
     jobFailed "$MODE requires 7 arguments invoke $0 with no arguments for more information"
  fi

  getFullPath $2
  declare -r TRAINING_DATA=$GETFULLPATHRET

  getFullPath $3
  declare -r INPUT_DATA=$GETFULLPATHRET

  getFullPath $4
  declare -r OUTPUT_DIR=$GETFULLPATHRET

  declare -r STAGE=$5
  declare -r LEVEL=$6
  declare -r BLOCKSIZE=$7

  runCreateMode
  THE_EXIT=$?

  writeJobProperties $OUTPUT_DIR

  if [ $THE_EXIT == 0 ] ; then
    logEcho ""
    logMessage "Next step is to do a test run by running the following command:"
    logEcho ""
    logMessage "$0 $TESTRUN_MODE $OUTPUT_DIR"
    logEcho ""
  fi

  logEcho ""
  logEndTime "$MODE mode" $MODE_START_TIME $THE_EXIT
  logEcho ""
  exit $THE_EXIT
fi


#
# Full run mode
#
if [ "$MODE" == "$FULLRUN_MODE" ] ; then

  if [ $# -ne 2 ] ; then
     jobFailed "$MODE requires 2 arguments invoke $0 with no arguments for more information"
  fi

  getFullPath $2

  declare -r OUTPUT_DIR=$GETFULLPATHRET

  getNumberOfCHMJobsFromConfig $OUTPUT_DIR
  if [ $? != 0 ] ; then
     jobFailed "Error obtaining number of jobs"
  fi

  getMatlabDirectory $OUTPUT_DIR
  if [ $? != 0 ] ; then
    jobFailed "Unable to get path to matlab directory"
  fi

  # Assume the first job has been run already, but we
  # will check anyways
  THE_START_JOB=2

  # Check that the first job was successful
  # if not set set THE_START_JOB to 1
  # Also check that is more then 1 job otherwise this does not
  # need to run
  checkSingleTask 1
  if [ $? != 0 ] ; then
     logMessage "Test job #1 appears to have failed.  Adding to run list"
     THE_START_JOB=1
  elif [ $NUMBER_JOBS -lt $THE_START_JOB ] ; then
     logMessage "There are only $NUMBER_JOBS job to run and it appears to have run successfully"
     logEndTime "$MODE mode" $MODE_START_TIME 0
     exit 0 
  fi
  
  runJobs "${THE_START_JOB}-${NUMBER_JOBS}"
  THE_EXIT=$?
  logEcho ""
  logEndTime "$MODE mode" $MODE_START_TIME $THE_EXIT

  exit $THE_EXIT
fi

#
# Print only mode
#
if [ "$MODE" == "$PRINTONLY_MODE" ] ; then

  if [ $# -ne 2 ] ; then
     jobFailed "$MODE requires 2 arguments invoke $0 with no arguments for more information"
  fi

  getFullPath $2
  declare -r OUTPUT_DIR=$GETFULLPATHRET
  echo $OUTPUT_DIR
  runPrintOnlyMode
  THE_EXIT=$?
  logEndTime "$MODE mode" $MODE_START_TIME $THE_EXIT
  exit $THE_EXIT
fi

#
# Test run mode
#
if [ "$MODE" == "$TESTRUN_MODE" ] ; then
  getFullPath $2
  declare -r OUTPUT_DIR=$GETFULLPATHRET

  getMatlabDirectory $OUTPUT_DIR
  if [ $? != 0 ] ; then
    jobFailed "Unable to get path to matlab directory"
  fi

  runJobs "1-1"
  THE_EXIT=$?
  
  if [ $THE_EXIT == 0 ] ; then

    
    getRunTimeOfSingleCHMTask $OUTPUT_DIR 1

    if [ $? == 0 ] ; then
       getNumberOfCHMJobsFromConfig $OUTPUT_DIR
       if [ $? == 0 ] ; then
          EST_HOURS=`echo "scale=0;($RUNTIME_SECONDS*$NUMBER_JOBS)/3600" | bc -l`
          logEcho ""
          logMessage "The test job took roughly $RUNTIME_SECONDS seconds to run"
          logMessage "The full run has $NUMBER_JOBS task(s) which means this job will take: $EST_HOURS hours of compute time"
          logEcho ""
  
       fi
    fi

    logEcho ""
    logMessage "Next step after checking results in $OUTPUT_DIR/out are satisfactory"
    logMessage "is to do a full run by running the following command:"
    logEcho ""
    logMessage "$0 $FULLRUN_MODE $OUTPUT_DIR"
    logEcho ""
  fi

  logEcho ""
  logEndTime "$MODE mode" $MODE_START_TIME $THE_EXIT
  exit $THE_EXIT
fi


#
# Verify mode
#
if [ "$MODE" == "$VERIFY_MODE" ] ; then
  getFullPath $2
  declare -r OUTPUT_DIR=$GETFULLPATHRET

  getNumberOfCHMJobsFromConfig $OUTPUT_DIR

  if [ $? != 0 ] ; then
     jobFailed "Error obtaining number of jobs"
  fi


  verifyResults 1 "1-${NUMBER_JOBS}" "print"
  VERIFY_EXIT=$?

  logEcho ""

  if [ $NUM_FAILED_JOBS -gt 0 ] ; then
    logMessage "Number failed jobs:  $NUM_FAILED_JOBS out of $NUMBER_JOBS task(s)"
  else 
    logMessage "All $NUMBER_JOBS task(s) completed successfully"
  fi
   
  logEndTime "$MODE mode" $MODE_START_TIME $VERIFY_EXIT
  exit $VERIFY_EXIT
fi

# implies an unsupported mode
jobFailed "$MODE not supported.  Invoke $0 for list of valid options."
