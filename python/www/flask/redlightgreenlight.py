from typing import List
from jetson_inference import poseNet
import numpy as np
import os
import threading
import sounddevice as sd
import soundfile as sf

BUFFER_SIZE = 10
MOVEMENT_THRESHOLD = 50
MIN_TIME_WAIT = 1
MAX_TIME_WAIT = 50
RED_TRANSITION_WAIT = 1
SOUND_DIR = "/home/rico/Documents/rlgl/"
SOUND_RED_LIGHT = "redlight.wav"
SOUND_GREEN_LIGHT = "greenlight.wav"
SOUND_GUN_SHOT = "gunshot.wav"
SOUND_INTRO = "intro.wav"


# Use print(sd.query_devices()) to find the output device index
SOUND_OUTPUT_DEVICE = 12


class RedLightGreenLight():
    
    def __init__(self):
        self.light_status_red = True
        self.sound_lock = threading.Lock()
        self.poses_avg = {}
        self.game_timer = 0 
        self.red_transition_wait = 0
        print(sd.query_devices())
        self.play_sound(SOUND_INTRO)
        self.pose_id_is_shot = {}
        
    
    def Process(self, poses: List[poseNet.ObjectPose.Keypoint]):
        
        #print(f"detected {len(poses)} objects pose in image")

        # Reset the timer
        self.game_timer -=1
        #print(f"Timer: {self.game_timer}")

        # Check if we should switch to red light
        if self.game_timer <= 0:
            # Change the light
            self.light_status_red = not self.light_status_red
            
            if self.light_status_red == True:
                print("Red Light")
                self.play_sound(SOUND_RED_LIGHT)
                self.red_transition_wait = RED_TRANSITION_WAIT
            else:
                print("Green Light")
                self.play_sound(SOUND_GREEN_LIGHT)
                self.clear_values()
            
            # Reset the timer to some random new time
            self.game_timer = np.random.randint(MIN_TIME_WAIT,MAX_TIME_WAIT)


        # Check if red light or green light
        if self.light_status_red == True:

            # Allow a little time to transition to red after hearing red light
            self.red_transition_wait -=1
            #print(f"Red Transition: {self.red_transition_wait}")
            
            movement_found = False
            
            if self.red_transition_wait < 0 :
                # If red, check for poses for movement            
                for pose in poses:
                    #print(pose)
                    #print(pose.Keypoints)
                    #print("Links", pose.Links)
                    
                    # If a shot has already been registered for this ID
                    # Move to the next pose
                    if pose.ID in self.pose_id_is_shot and self.pose_id_is_shot[pose.ID] == True:
                        print(f"{pose.ID} already shot")
                        continue
                    
                    # Check for movement with the current pose
                    movement_found = self.check_pose(pose)                 
                    if movement_found:
                        self.pose_id_is_shot[pose.ID] = True
                    
                    # Average pose to track only large movement
                    self.avg_pose(pose)
                    
                if movement_found:
                    self.play_sound(SOUND_GUN_SHOT)
                            
        
    def check_pose(self, pose: poseNet.ObjectPose.Keypoint):
        """
        Check the pose against the average.  If there is movement found
        based on the threshold then set the flag of movement found.
        """
        movement_found = False
        
        for keypoint in pose.Keypoints:
            pose_key = f"{pose.ID}-{keypoint.ID}"      

            if pose_key in self.poses_avg:
                avg_x = self.poses_avg[pose_key]["avg_x"]
                avg_y = self.poses_avg[pose_key]["avg_y"]
                
                if abs(avg_x - keypoint.x) > MOVEMENT_THRESHOLD:
                    print(print(f"X Movement: {pose_key} {avg_x} {keypoint.x}"))
                    movement_found = True
                    break
                
                if abs(avg_y - keypoint.y) > MOVEMENT_THRESHOLD:
                    print(print(f"Y Movement: {pose_key} {avg_y} {keypoint.y}"))
                    movement_found = True
                    break
    
        return movement_found
    
    def avg_pose(self, pose: poseNet.ObjectPose.Keypoint):
        """
        Accumulate and average the pose.  This will take all the
        keypoints and average there x and y value.  The average
        is based on the number of accumulated values.
        """
        for keypoint in pose.Keypoints:
            pose_key = f"{pose.ID}-{keypoint.ID}"             
            if pose_key not in self.poses_avg:                
                self.poses_avg[pose_key] = {
                    "accum_x": [keypoint.x],
                    "accum_y": [keypoint.y],
                    "avg_x": keypoint.x,
                    "avg_y": keypoint.y
                }
            else:
                self.poses_avg[pose_key]["accum_x"].append(keypoint.x)
                self.poses_avg[pose_key]["accum_y"].append(keypoint.y)
                
                if len(self.poses_avg[pose_key]["accum_x"]) > BUFFER_SIZE:
                    # Accumulate only last BUFFER_SIZE
                    self.poses_avg[pose_key]["accum_x"].pop(0)
                    self.poses_avg[pose_key]["accum_y"].pop(0)
                    
                    # Get the average
                    self.poses_avg[pose_key]["avg_x"] = np.mean(self.poses_avg[pose_key]["accum_x"]) 
                    self.poses_avg[pose_key]["avg_y"] = np.mean(self.poses_avg[pose_key]["accum_y"])

                    #print(f"{pose_key} {averages}")
                    #print(averages)
                    
    def clear_values(self):
        self.poses_avg = {}
        self.pose_id_is_shot = {}
    
    def play_sound_in_thread(self, sound_name: str, sound_dir: str = SOUND_DIR):
        thread = threading.Thread(target=self.play_sound_in_thread, args=(sound_name, sound_dir))
        thread.start()
    
    def play_sound(self, sound_name: str, sound_dir: str = SOUND_DIR):

        with self.sound_lock:
            # Set the file name to play        
            filename = os.path.join(sound_dir, sound_name)

            try:
                # Read the WAV file
                data, fs = sf.read(filename, dtype='float32')
                
                # Play the WAV file on the specified output device
                sd.play(data, fs, device=SOUND_OUTPUT_DEVICE)
                
                # Wait until playback finishes
                sd.wait()
                
                print(f"{sound_name} Playback finished.")
            except Exception as e:
                print(f"An error occurred during playback {sound_name}: {e}")
        