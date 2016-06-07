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
import org.opencv.features2d.DescriptorExtractor; // Surf Extractor
import org.opencv.features2d.DescriptorMatcher; // Surf Feature Matcher
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
// Surf feature extractor
DescriptorExtractor surfExtractor;
// Surf feature matcher for correspondence
DescriptorMatcher surfMatcher;


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

//
PImage gridtex = loadImage("/Users/rajarshiroy/Desktop/CS231a/Project/implementation/grid_tex.jpg");


void setup() {
  size(1400, 900, P3D);
  //size(1440, 900, P3D);
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
  surfExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
  surfMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_L1);
  blobDetector.read("/Users/rajarshiroy/Desktop/CS231a/Project/implementation/simpleblob_custom_params.yml");
  surfExtractor.read("/Users/rajarshiroy/Desktop/CS231a/Project/implementation/surf_custom_params.yml");
  smooth();
  
}
 
void draw() {
  if (leap.hasImages()) {
    leapInit = true;
    for (Image camera : leap.getImages()) {
      if (camera.isLeft()) {
        // Left camera
        //image(camera, 0, 0);
        leftCam = camera;
      } else {
        // Right camera
        //image(camera, 0, camera.getHeight());
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
  //System.out.println("Detected " + blobs.size()+ " blobs in the image");



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
       //System.out.println("x: " + x + " y:" + y);
     }
  }
  
  //System.out.println("triEpiFiltsize: " + triEpiFilt.size());
  
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
  


  
  
  background(0);
  
  
  // ========== DRAW INFORMATION START==========
  // Draw the camera images
  srcLeft.resize(320, 240);
  srcRight.resize(320, 240);
  image(srcLeft, 0, 0);
  image(srcRight, 320, 0);
 
  // Draw unrectified blobs
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
  stroke(150);
  for (PVector blobslope : leftSlopesEpifilt) {
    fill(0, 0, 0);
    rect(blobslope.x*80+160, blobslope.y*80+360-2, 16, 4);
    fill(255, 200, 0);
    ellipse(blobslope.x*80+160, blobslope.y*80+360, 5, 5);
    
    //System.out.println("X: " + blobslope.x+ "Y: " + blobslope.y);
  }
  
  for (PVector blobslope : rightSlopesEpifilt) {
    fill(255, 100, 0);
    ellipse(blobslope.x*80+160, blobslope.y*80+360, 5, 5);
    //System.out.println("X: " + blobslope.x+ "Y: " + blobslope.y);
  }
  
  
  // Draw rectified markers together
  fill(100, 250, 0);
  ellipse(leftCamLeftMarker.x*80+480, leftCamLeftMarker.y*80+360, 5, 5);
  ellipse(rightCamLeftMarker.x*80+480, rightCamLeftMarker.y*80+360, 5, 5);
  fill(100, 0, 250);  
  ellipse(leftCamRightMarker.x*80+480, leftCamRightMarker.y*80+360, 5, 5);
  ellipse(rightCamRightMarker.x*80+480, rightCamRightMarker.y*80+360, 5, 5);
  
  // Draw a box
  pushMatrix();
  fill(100, 100, 100);
  stroke(50);
  translate(width/2, height/2, 0);
  scale(0.5);
    box(40, 20, 10);

    pushMatrix();
    translate(0, 98, 0);
    box(290, 175, 2);
    popMatrix();
    
    pushMatrix();
    translate(leftMarkerPos.x, leftMarkerPos.y, leftMarkerPos.z);
    fill(200, 250, 0);
    sphere(20);
    popMatrix();
  
    pushMatrix();
    translate(rightMarkerPos.x, rightMarkerPos.y, rightMarkerPos.z);
    fill(200, 0, 250);
    sphere(20);
    popMatrix();
    
  
  
  popMatrix();

  // Draw framerate
  fill(0);
  noStroke();
  rect(0, 0, 110, 30);
  fill(255);
  //text("Frame rate: " + nf(round(frameRate), 2), 10, 20, 0);
  // ========== DRAW INFORMATION END==========

  
  leftImage.release();
  rightImage.release();
  
  //drawRoom();
  //drawFancyRoom();
  
}
 

void drawRoom(){
  //Width: 1440 2D res corresponds to 290
  //Height: 900 2D res corresponds to 175
  //~130 in leap 3D coord corresponds to 140mm in reality
  
  // Camera position in pixel coords 
  float camx = (leftMarkerPos.x+rightMarkerPos.x)/2*5.385+720;
  float camy = (leftMarkerPos.y+rightMarkerPos.y)/2*5.385+490;
  float camz = (leftMarkerPos.z+rightMarkerPos.z)/2*5.385;
  
  // Far wall distance in pixel coords
  float d = 2000;
  
  float xAC = camx*d/camz;
  float xBD = 1440-((1440-camx)*d/camz);
  float yAB = camy*d/camz;
  float yCD = 900-((900-camy)*d/camz);
  
  stroke(255);
  strokeWeight(4);
  // Line CD
  line(xAC, yCD, 0, xBD, yCD, 0);
  // Line AC
  line(xAC, yAB, 0, xAC, yCD, 0);
  // Line BD
  line(xBD, yAB, 0, xBD, yCD, 0);
  // Line AB
  line(xAC, yAB, 0, xBD, yAB, 0);
  
  // Line Aa
  line(xAC, yAB, 0, 0, 0, 0);
  // Line Bb
  line(xBD, yAB, 0, 1440, 0, 0);
  // Line Cc
  line(xAC, yCD, 0, 0, 900, 0);
  // Line Dd
  line(xBD, yCD, 0, 1440, 900, 0);
  
}

void drawFancyRoom(){
  //Width: 1440 2D res corresponds to 290
  //Height: 900 2D res corresponds to 175
  //~130 in leap 3D coord corresponds to 140mm in reality
  
  // Camera position in pixel coords 
  float camx = (leftMarkerPosAvg.x+rightMarkerPosAvg.x)/2*5.385+720;
  float camy = (leftMarkerPosAvg.y+rightMarkerPosAvg.y)/2*5.385+490;
  float camz = (leftMarkerPosAvg.z+rightMarkerPosAvg.z)/2*5.385;
  
  // Far wall distance in pixel coords
  float d = 2000;
  
  float xAC = camx*d/camz;
  float xBD = 1440-((1440-camx)*d/camz);
  float yAB = camy*d/camz;
  float yCD = 900-((900-camy)*d/camz);
  
  // Ceiling
  beginShape();
  texture(gridtex);
  vertex(0, 0, 0, 0); //a
  vertex(1400, 0, 350, 0); //b
  vertex(1400, 500, 350, 350); //B  xBD, yAB,
  vertex(0, 900, 0, 350); //A  xAC, yAB,
  endShape();
  
  stroke(255);
  strokeWeight(4);
  // Line CD
  line(xAC, yCD, 0, xBD, yCD, 0);
  // Line AC
  line(xAC, yAB, 0, xAC, yCD, 0);
  // Line BD
  line(xBD, yAB, 0, xBD, yCD, 0);
  // Line AB
  line(xAC, yAB, 0, xBD, yAB, 0);
  
  // Line Aa
  line(xAC, yAB, 0, 0, 0, 0);
  // Line Bb
  line(xBD, yAB, 0, 1440, 0, 0);
  // Line Cc
  line(xAC, yCD, 0, 0, 900, 0);
  // Line Dd
  line(xBD, yCD, 0, 1440, 900, 0);
  
}


boolean sketchFullScreen() {
  //return true;
  return false;
}


void captureEvent(Capture _c) {
  _c.read();
}
