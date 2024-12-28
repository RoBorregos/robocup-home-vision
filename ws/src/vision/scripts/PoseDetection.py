#!/usr/bin/env python3
# import rospy
import cv2
import mediapipe as mp
import numpy as np

import os # For testing

'''
Questions:
- Should this be a ROS node or is it fine as a standalone script?
'''

class PoseDetection:
    def __init__(self):
        print("Pose Detection Ready")
        # rospy.init_node('pose_detection')
        # rospy.loginfo("Pose Detection Ready")

        # Initialize MediaPipe Pose as a class attribute
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils  # For visualizing landmarks

    def detectPose(self):
        pass

    def detectGesture(self):
        pass

    def detectClothes(self):
        pass

    def isChestVisible(self, image_path):
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.pose.process(image_rgb)

        # Check for landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key points for shoulders and chest (sternum approximate region)
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Check visibility and positioning
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                print("Chest is visible.")
                return True
            else:
                print("Chest is not fully visible.")
        else:
            print("No pose detected.")
        return False

    def chestPosition(self, image_path, save_image=False):
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image.")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect pose landmarks
        results = self.pose.process(image_rgb)

        # Check for landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Approximate chest region as below the nose and between shoulders
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * image.shape[1])
                chest_y = int((left_shoulder.y + right_shoulder.y) / 2 * image.shape[0])

                # Save the image with the chest center marked
                if save_image:
                    # Draw a circle at the approximated chest position
                    cv2.circle(image, (chest_x, chest_y), 10, (255, 0, 0), -1)
                    cv2.imwrite("./testImages/chest_position.jpg", image)

                return (chest_x, chest_y)

        print("Chest landmarks not detected or not fully visible.")
        return None
    
    def personAngle(self, image_path):
        pass

def main():
    # image_path = "./testImages/image4.jpg"

    # pose_detection = PoseDetection()

    # print(pose_detection.isChestVisible(image_path=image_path))

    # chest_coords = pose_detection.chestPosition(image_path=image_path, save_image=True)
    # if chest_coords:
    #     print(f"Chest coordinates: {chest_coords}")

    # angle = pose_detection.personAngle(image_path=image_path)
    # if angle:
    #     print(f"Person angle: {angle:.2f} degrees")

    test_images_dir = "./testImages/helicopterhelicopter"
    pose_detection = PoseDetection()

    for i in range(1, 20):
        image_name = f"{i}.jpeg"
        image_path = os.path.join(test_images_dir, image_name)
        angle = pose_detection.personAngle(image_path=image_path)
        print(angle)

if __name__ == '__main__':
    main()
