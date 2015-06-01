#!/bin/bash

if [ $# -ne 3 ] ; then
   echo "$0 <remote deploy host> <remote base dir> <properties file>"
   echo ""
   echo "This script packages and deploys panfishCHM to <remote deploy host>"
   echo "using ssh/scp commands installing the package to <remote base dir>"
   echo "directory. The <properties file> is copied over as "
   echo "panfishCHM.properties file"
   exit 1
fi
HOST=$1
DEPLOY_BASE_DIR=$2
PROPERTIES_FILE=$3


# we are going to assume panfish Makefile.pl is in the parent directory to where the script
# is located.
PANFISHCHM_DIR="`dirname $0`"

TIMESTAMP=`date +%m.%d.%Y.%H.%M.%S`

INSTALL_DIR_NAME="panfishCHM.${TIMESTAMP}"

TMPINSTALLDIR="/tmp/$INSTALL_DIR_NAME"

cd $PANFISHCHM_DIR

# create tmp directory
/bin/mkdir -p $TMPINSTALLDIR

if [ $? != 0 ] ; then
  echo "Error unable to create $TMPINSTALLDIR"
  exit 1
fi

#copy over createCHMJob.sh
/bin/cp createCHMJob.sh $TMPINSTALLDIR/.
if [ $? != 0 ] ; then
  echo "Error unable to run /bin/cp createCHMJob.sh $TMPINSTALLDIR/."
  exit 1
fi

chmod a+x $TMPINSTALLDIR/createCHMJob.sh
if [ $? != 0 ] ; then
  echo "Error unable to run chmod a+x $TMPINSTALLDIR/createCHMJob.sh"
  exit 1
fi

# copy over createCHMTrainJob.sh
/bin/cp createCHMTrainJob.sh $TMPINSTALLDIR/.
if [ $? != 0 ] ; then
  echo "Error unable to run /bin/cp createCHMTrainJob.sh $TMPINSTALLDIR/."
  exit 1
fi

chmod a+x $TMPINSTALLDIR/createCHMTrainJob.sh
if [ $? != 0 ] ; then
  echo "Error unable to run chmod a+x $TMPINSTALLDIR/createCHMTrainJob.sh"
  exit 1
fi

# copy over scripts folder
/bin/cp -a scripts $TMPINSTALLDIR/.

if [ $? != 0 ] ; then
  echo "Error unable to run /bin/cp -a scripts $TMPINSTALLDIR/."
  exit 1
fi

# copy over config file into temp directory
if [ ! -s "$PROPERTIES_FILE" ] ; then
   echo "$PROPERTIES_FILE properties file not found"
   exit 1
fi


/bin/cp $PROPERTIES_FILE $TMPINSTALLDIR/panfishCHM.properties

if [ $? != 0 ] ; then
  echo "Unable to run /bin/cp $PROPERTIES_FILE $TMPINSTALLDIR/panfishCHM.properties"
  exit 1
fi

SCP_ARG="${HOST}:${DEPLOY_BASE_DIR}/."

# copy up new version set folder name to date timestamp
scp -r $TMPINSTALLDIR $SCP_ARG

if [ $? != 0 ] ; then
  echo "Error running scp -r $TMPINSTALLDIR $SCP_ARG"
  exit 1
fi


# change symlink by removing first then creating
ssh $HOST "/bin/rm $DEPLOY_BASE_DIR/panfishCHM"

if [ $? != 0 ] ; then
  echo "Error running ssh $HOST \"/bin/rm $DEPLOY_BASE_DIR/panfishCHM\""
  exit 1
fi


ssh $HOST "/bin/ln -s $DEPLOY_BASE_DIR/$INSTALL_DIR_NAME  $DEPLOY_BASE_DIR/panfishCHM"

if [ $? != 0 ] ; then
  echo "Error running ssh $HOST \"/bin/ln -s $DEPLOY_BASE_DIR/$INSTALL_DIR_NAME  $DEPLOY_BASE_DIR/panfishCHM\""
  exit 1
fi

echo "Removing $TMPINSTALLDIR"
/bin/rm -rf $TMPINSTALLDIR

# exit
