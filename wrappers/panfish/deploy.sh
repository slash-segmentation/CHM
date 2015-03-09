#!/bin/bash

if [ $# -ne 1 ] ; then
   echo "$0 <environment (should be prefix to properties file and .templates dir under properties/ directory>"
   exit 1
fi

DEPLOY_ENV=$1

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

PROPERTIES_FILE="properties/${DEPLOY_ENV}.properties"

if [ ! -s "$PROPERTIES_FILE" ] ; then
   echo "$DEPLOY_ENV is missing $PANFISHCHM_DIR/$PROPERTIES_FILE file"
   exit 1
fi


/bin/cp $PROPERTIES_FILE $TMPINSTALLDIR/panfishCHM.properties

if [ $? != 0 ] ; then
  echo "Unable to run /bin/cp $PROPERTIES_FILE $TMPINSTALLDIR/panfishCHM.properties"
  exit 1
fi



HOST="NOTSET"
SCP_ARG="NOTSET"

if [ "$DEPLOY_ENV" == "local" ] ; then
  /bin/mv $TMPINSTALLDIR panfishCHM
  if [ $? != 0 ] ; then
    echo "Error running /bin/mv $TMPINSTALLDIR panfishCHM"
    exit 1
  fi   
  exit 0
fi

if [ "$DEPLOY_ENV" == "coleslaw" ] ; then
   HOST="churas@localhost"
   DEPLOY_BASE_DIR="/home/churas/bin"
   SCP_ARG="${HOST}:${DEPLOY_BASE_DIR}/."
fi

if [ "$DEPLOY_ENV" == "prod" ] ; then
   HOST="tomcat@cylume.camera.calit2.net"
   DEPLOY_BASE_DIR="/home/validation/camera/release/ncmir/bin"
   SCP_ARG="${HOST}:${DEPLOY_BASE_DIR}/."
fi

if [ "$DEPLOY_ENV" == "megashark" ] ; then
   HOST="megashark.crbs.ucsd.edu"
   DEPLOY_BASE_DIR="/sharktopus/megashark/cws/bin"
   SCP_ARG="${HOST}:${DEPLOY_BASE_DIR}/."
fi

if [ "$DEPLOY_ENV" == "vizwall" ] ; then
   HOST="churas@137.110.119.214"
   DEPLOY_BASE_DIR="/home/churas/panfish/cws_vizwall/cws/bin"
   SCP_ARG="${HOST}:${DEPLOY_BASE_DIR}/."
fi

if [ "$HOST" == "NOTSET" ] ; then
  echo "Please setup $DEPLOY_ENV in this script $0"
  exit 1
fi

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
