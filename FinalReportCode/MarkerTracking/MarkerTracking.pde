////////////////////////////////////////////////////////////////////////
// Performs realtime 3D marker tracking of retro-reflective markers
// on user's eyewear using Leap Motion controller stereo camera stream.
////////////////////////////////////////////////////////////////////////

import processing.video.*;
import de.voidplus.leapmotion.*;
import gab.opencv.*;
import org.opencv.video.Video;
import org.opencv.core.Mat;
import org.opencv.features2d.DMatch;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint; // SimpleBlob
import org.opencv.imgproc.Imgproc; // HoughCircles
import org.opencv.features2d.FeatureDetector; // SimpleBlob
import org.opencv.features2d.KeyPoint; // SimpleBlob
import java.util.List;
import java.util.LinkedList;

//Capture cap;
LeapMotion leap;
Image leftCam, rightCam;
boolean leapInit = false;
OpenCV leftcvUnstretched, rightcvUnstretched, leftcv, rightcv;
PImage srcLeft, srcRight;

// Blob detector
FeatureDetector blobDetector;

// Rectified image plane marker position
PVector leftCamLeftMarker = new PVector(0,0,0);
PVector leftCamRightMarker = new PVector(0,0,0);
PVector rightCamLeftMarker = new PVector(0,0,0);
PVector rightCamRightMarker= new PVector(0,0,0);
// Triangulated 3D marker position
PVector leftMarkerPos = new PVector(0,0,0);
PVector rightMarkerPos = new PVector(0,0,0);

// Moving average queues for smoothing
LinkedList<PVector> leftMarkerPosQueue;
LinkedList<PVector> rightMarkerPosQueue;
PVector leftMarkerPosAvg = new PVector(0,0,0);
PVector rightMarkerPosAvg = new PVector(0,0,0);


void setup() {
  //size(1400, 900, P3D);
  size(1440, 900, P3D);
  background(0);
  leap = new LeapMotion(this);
  leftcvUnstretched = new OpenCV(this, 640, 240);
  rightcvUnstretched = new OpenCV(this, 640, 240);
  leftcv = new OpenCV(this, 640, 480);
  rightcv = new OpenCV(this, 640, 480);
  leftMarkerPosQueue = new LinkedList<PVector>();
  rightMarkerPosQueue = new LinkedList<PVector>();
  leftMarkerPosQueue.add(new PVector(0,0,0));
  leftMarkerPosQueue.add(new PVector(0,0,0));
  leftMarkerPosQueue.add(new PVector(0,0,0));
  rightMarkerPosQueue.add(new PVector(0,0,0));
  rightMarkerPosQueue.add(new PVector(0,0,0));
  rightMarkerPosQueue.add(new PVector(0,0,0));
  
  blobDetector = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
  blobDetector.read("/Users/rajarshiroy/Desktop/CS231a/Project/implementation/simpleblob_custom_params.yml");
  smooth();
  
}
 
void draw() {
  if (leap.hasImages()) {
    leapInit = true;
    for (Image camera : leap.getImages()) {
      if (camera.isLeft()) {
        leftCam = camera;
      } else {
        rightCam = camera;
      }
    }
  }
  // If first images not yet captured then skip
  // Otherwise null pointer at leftCam, rightCam
  if(!leapInit) return;
  
  // Stretch images vertically*2 from 640*240->640*480
  leftcvUnstretched.loadImage(leftCam);
  rightcvUnstretched.loadImage(rightCam);
  PImage leftCamStretched = leftcvUnstretched.getSnapshot();
  PImage rightCamStretched = rightcvUnstretched.getSnapshot();
  leftCamStretched.resize(640, 480);
  rightCamStretched.resize(640,480);
  leftcv.loadImage(leftCamStretched);
  rightcv.loadImage(rightCamStretched);
  
  // Get OpenCV Matrices of Left and Right images
  // Output: leftImage, rightImage Mat
  srcLeft = leftcv.getSnapshot();   //src is PImage type
  srcRight = rightcv.getSnapshot();
  leftcv.gray();
  rightcv.gray();
  Mat leftImage = leftcv.getGray();
  Mat rightImage = rightcv.getGray();
  
  // Detect Blobs in Left and Right Images
  // Output: blobsLeft, blobsRight List<KeyPoint>
  MatOfKeyPoint blobMatLeft = new MatOfKeyPoint();
  blobDetector.detect(leftImage, blobMatLeft);
  List<KeyPoint> blobsLeft = blobMatLeft.toList();
  MatOfKeyPoint blobMatRight = new MatOfKeyPoint();
  blobDetector.detect(rightImage, blobMatRight);
  List<KeyPoint> blobsRight = blobMatRight.toList();


  // Rectify blobsLeft and blobsRight
  ArrayList<PVector> leftSlopes = new ArrayList<PVector>();
  ArrayList<PVector> rightSlopes = new ArrayList<PVector>();
  com.leapmotion.leap.Image leftCamRectifier = leftCam.getRaw();
  com.leapmotion.leap.Image rightCamRectifier = rightCam.getRaw();
  
  if(blobsLeft.size()>0) {
    for (int i=0; i<blobsLeft.size(); i++) {
      KeyPoint blob = blobsLeft.get(i);
      com.leapmotion.leap.Vector blobvector = new com.leapmotion.leap.Vector((float)blob.pt.x, (float)blob.pt.y/2, 0);
      com.leapmotion.leap.Vector blobslope = leftCamRectifier.rectify(blobvector);
      leftSlopes.add(new PVector(blobslope.get(0), blobslope.get(1), 0));
    }
  }
  
  if(blobsRight.size()>0) {
    for (int i=0; i<blobsRight.size(); i++) {
      KeyPoint blob = blobsRight.get(i);
      com.leapmotion.leap.Vector blobvector = new com.leapmotion.leap.Vector((float)blob.pt.x, (float)blob.pt.y/2, 0);
      com.leapmotion.leap.Vector blobslope = rightCamRectifier.rectify(blobvector);
      rightSlopes.add(new PVector(blobslope.get(0), blobslope.get(1), 0));
    }
  }

  
  // Make a epipolar constraint filtered match list
  ArrayList<PVector> leftSlopesEpifilt = new ArrayList<PVector>();
  ArrayList<PVector> rightSlopesEpifilt = new ArrayList<PVector>();
  for (PVector leftSlope : leftSlopes) {
    for (PVector rightSlope : rightSlopes) {
      if ( ((leftSlope.x-rightSlope.x)<0) && ((rightSlope.x-leftSlope.x)<0.2) && 
           ((leftSlope.y-rightSlope.y)<0.025) && ((rightSlope.y-leftSlope.y)<0.025) ) {
        leftSlopesEpifilt.add(leftSlope);
        rightSlopesEpifilt.add(rightSlope);
      }
    }
  }
  
  
  // Triangulate filtered list
  ArrayList<PVector> triEpiFilt = new ArrayList<PVector>();
  for (int i=0; i<leftSlopesEpifilt.size(); i++) {
     PVector leftPoint = leftSlopesEpifilt.get(i);
     PVector rightPoint = rightSlopesEpifilt.get(i);
     float z = 40/(rightPoint.x - leftPoint.x);
     float y = z * (rightPoint.y+leftPoint.y) / 2;
     float x = 20 - z * (rightPoint.x+leftPoint.x) / 2;
     if( (z>200) && (z<700) && (x>-500) && (x<500) && (y>-300) && (y<300)){
       triEpiFilt.add(new PVector(x,y,z));
     }
  }
  
    
  // Further filtering if excess markers detected
  if(triEpiFilt.size()==0) {
    // No markers detected: don't update either
  } else if(triEpiFilt.size()==1) {
    // One marker detected: don't update either (for now)
  } else if(triEpiFilt.size()==2) {
    // Two markers detected: left marker has smaller x
    PVector temp0 = triEpiFilt.get(0);
    PVector temp1 = triEpiFilt.get(1);
    if(temp0.x < temp1.x) {
      leftMarkerPos = temp0;
      rightMarkerPos = temp1;
    } else {
      leftMarkerPos = temp1;
      rightMarkerPos = temp0;
    }
  } else {
    // More than two markers detected
    // Find the two separated by real world marker distance:
    // 115 to 145
    for (PVector temp0 : triEpiFilt) {
      for (PVector temp1 : triEpiFilt) {
        if( (temp0.dist(temp1)>115) && (temp0.dist(temp1)<145) ) {
          if(temp0.x < temp1.x) {
            leftMarkerPos = temp0;
            rightMarkerPos = temp1;
          } else {
            leftMarkerPos = temp1;
            rightMarkerPos = temp0;
          }
        }
      }
    }
  }
  
  System.out.println("distance: " + rightMarkerPos.dist(leftMarkerPos));

  // Smoothing via moving average
  leftMarkerPosQueue.add(leftMarkerPos);
  leftMarkerPosQueue.removeFirst();
  rightMarkerPosQueue.add(rightMarkerPos);
  rightMarkerPosQueue.removeFirst();
  
  leftMarkerPosAvg = new PVector(0,0,0);
  rightMarkerPosAvg = new PVector(0,0,0);
  for (PVector temp : leftMarkerPosQueue) {
    leftMarkerPosAvg.add(temp);
  }
  for (PVector temp : rightMarkerPosQueue) {
    rightMarkerPosAvg.add(temp);
  }  
  leftMarkerPosAvg.div(leftMarkerPosQueue.size());
  rightMarkerPosAvg.div(rightMarkerPosQueue.size());
  
  
  // ========== DRAW INFORMATION START==========
  background(0);
  
  // Draw the raw camera images
  srcLeft.resize(320, 240);
  srcRight.resize(320, 240);
  image(srcLeft, 0, 0);
  image(srcRight, 320, 0);
 
  // Draw unrectified blobs on top of raw images
  fill(255, 200, 0);
  if(blobsLeft.size()>0) {
    for (int i=0; i<blobsLeft.size(); i++) {
      KeyPoint blob = blobsLeft.get(i);
      ellipse((float)blob.pt.x/2, (float)blob.pt.y/2, 3, 3);
    } 
  }
  fill(255, 100, 0);
  if(blobsRight.size()>0) {
    for (int i=0; i<blobsRight.size(); i++) {
      KeyPoint blob = blobsRight.get(i);
      ellipse((float)blob.pt.x/2+320, (float)blob.pt.y/2, 3, 3);
    } 
  }
  
  // Draw all rectified blobs together (orange left, yellow right)
  // Also draw epipolar constraint triangle
  for (PVector blobslope : leftSlopesEpifilt) {
    stroke(255,50,0);
    fill(0, 0, 0);
    rect(blobslope.x*80+160, blobslope.y*80+360-2, 16, 4);
    noStroke();
    fill(255, 200, 0);
    ellipse(blobslope.x*80+160, blobslope.y*80+360, 5, 5);
  }
  for (PVector blobslope : rightSlopesEpifilt) {
    fill(255, 100, 0);
    ellipse(blobslope.x*80+160, blobslope.y*80+360, 5, 5);
  }
  
  
  pushMatrix();
  fill(100, 100, 100);
  stroke(50);
  translate(width/2, height/2, 0);
  scale(0.75);
    // Draw Leap Motion Controller
    box(40, 20, 10);
    // Draw Monitor
    pushMatrix();
    translate(0, 98, 0);
    box(290, 175, 2);
    popMatrix();
    // Draw Left Marker
    pushMatrix();
    translate(leftMarkerPos.x, leftMarkerPos.y, leftMarkerPos.z);
    fill(200, 250, 0);
    sphere(20);
    popMatrix();
    // Draw Right Marker
    pushMatrix();
    translate(rightMarkerPos.x, rightMarkerPos.y, rightMarkerPos.z);
    fill(200, 0, 250);
    sphere(20);
    popMatrix();
  popMatrix();
  // ========== DRAW INFORMATION END==========
  
  leftImage.release();
  rightImage.release();
}
 
boolean sketchFullScreen() {
  return true;
  //return false;
}

void captureEvent(Capture _c) {
  _c.read();
}
