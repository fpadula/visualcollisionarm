using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArmController : MonoBehaviour{

    public enum RAxis{
        X,
        Y,
        Z,
        Minus_X,
        Minus_Y,
        Minus_Z
    }

    public bool physics_enabled, verbose, agent_control, joints_reached_target_position, ignore_collisions, ignore_sol_dist;    
    public Transform StartingPose, FollowTarget;
    public Vector3 TargetOffset;
    public Transform Base;
    public Transform[] JointTransforms;
    public float[] JointAngles;
    public RAxis[] RotationAxis;    
    public int no_of_big_jumps, MaxStepsBeforeTimeout;
    
    private InverseKinematics ik;
    private ManipulatorJoint[] MJoints;    
    private bool reset_pose, joint_timed_out;
    public int time_out_counter;
    private float[] JointBuffer;

    void Start(){
        this.JointAngles = new float[JointTransforms.Length];                 
        this.JointBuffer = new float[JointTransforms.Length];                 

        if(this.physics_enabled){
            this.MJoints = new ManipulatorJoint[7];
            for(int i = 0; i < 7; i++){
                this.MJoints[i] = this.JointTransforms[i].GetComponent<ManipulatorJoint>();
            }
        }     
        this.ik = GetComponent<InverseKinematics>();
        this.reset_pose = false;
        // ResetPose();
    }

    private float MapValue(float value, float from1, float to1, float from2, float to2) {
        return value*((to1 - to2)/(from1 - from2)) + (-to1*from2 + from1 * to2)/(from1 - from2);
    }

    public Vector3 GetRandomValidPos(){
        // string angles = "";
        Vector3 pos;
        int layerMask = 1 << 11 | 1 << 8;        
        for(int i = 0; i < JointTransforms.Length; i++){
            this.JointBuffer[i] = MapValue(UnityEngine.Random.value, 0.0f, this.MJoints[i].GetMinLim(), 1.0f, this.MJoints[i].GetMaxLim());
            // Debug.Log(this.MJoints[i].GetMinLim() + ", " + this.MJoints[i].GetMaxLim());
            // angles += this.JointBuffer[i] + ", ";            
        }
        this.JointBuffer[2] = 0.0f;
        // Debug.Log(angles);
        pos = this.ik.ComputeFk(this.JointBuffer, false);
        if ((pos.y <= 0.25f) || Physics.CheckSphere(this.transform.parent.TransformPoint(pos), 0.1f, layerMask)){
            // Debug.Log("Invalid position, retrying...");
            return GetRandomValidPos();
        }
        return pos;
    }

    public Vector3 GetJointPos(int joint_index, bool in_word_coords){
        if(in_word_coords)
            return this.JointTransforms[joint_index].position;
        else
            return new Vector3(this.JointTransforms[joint_index].localPosition.x, this.JointTransforms[joint_index].localPosition.z, -this.JointTransforms[joint_index].localPosition.y);
    }
    public Vector3 GetJointPos(int joint_index){ return GetJointPos(joint_index, false);}    

    public Quaternion GetJointRot(int joint_index, bool in_word_coords){
        return in_word_coords? this.JointTransforms[joint_index].rotation : this.JointTransforms[joint_index].localRotation;   
    }
    public Quaternion GetJointRot(int joint_index){ return GetJointRot(joint_index, false);}       

    public void SetMaxJointSolDist(double size){ this.ik.max_joint_dist = size;}
    public void SetMaxCartSolDist(double size){ this.ik.max_cart_dist = size;}

    void SetJointsPositions(float[] angles){
        Vector3 eAngles;
        this.joints_reached_target_position = false;
        this.joint_timed_out = false;
        this.time_out_counter = 0;
        for(int i = 0; i < angles.Length; i++){
            if(!physics_enabled){
                if(RotationAxis[i] == RAxis.X)
                    eAngles = new Vector3(angles[i], 0, 0);
                else if(RotationAxis[i] == RAxis.Minus_X)
                    eAngles = new Vector3(-angles[i], 0, 0);
                else if(RotationAxis[i] == RAxis.Y)
                    eAngles = new Vector3(0, angles[i], 0);
                else if(RotationAxis[i] == RAxis.Minus_Y)
                    eAngles = new Vector3(0, -angles[i], 0);
                else if(RotationAxis[i] == RAxis.Z)
                    eAngles = new Vector3(0, 0, angles[i]);
                else
                    eAngles = new Vector3(angles[i], 0, -angles[i]);
                // JointBuffer[i] = Mathf.Deg2Rad*angles[i];
                // current_ja[i] = System.Math.Round(current_ja[i], no_digits);
                JointTransforms[i].localEulerAngles = eAngles;
                this.joints_reached_target_position = true;
            }
            else{
                this.MJoints[i].SetJointAngle(angles[i]);
            }
        }         
    }
    public bool testFk;
    // Update is called once per frame
    void Update(){
        if(this.testFk){
            SetJointsPositions(JointAngles);
            Vector3 fkpos = this.ik.ComputeFk(JointAngles, false);
            // Debug.Log(fkpos);
            this.FollowTarget.localPosition = fkpos;
        }
        else{
            if((!agent_control) && (FollowTarget != null)){
                // SetEEPose(FollowTarget.localPosition + FollowTarget.rotation*TargetOffset, FollowTarget.localRotation);
                SetEEPose(FollowTarget.localPosition + FollowTarget.rotation*TargetOffset, FollowTarget.localRotation, ignore_sol_dist);             
            }
            // else{            
            //     SetJointsPositions(JointAngles);            
            // }
        }
    }

    public int SetEEPose(Vector3 position, Quaternion orientation, bool ignore_sol_dist){
        int no_of_sols;
        this.ik.UpdateJointBuffer(this.JointAngles);
        no_of_sols = this.ik.ComputeIK(position, orientation, ignore_sol_dist);
        // if(no_of_sols == 0){
        //     no_of_sols = this.ik.ComputeIK(position, orientation, true);
        // }
        if(no_of_sols != 0){
            this.ik.UpdateJointAngles(this.JointAngles, false);
            SetJointsPositions(this.JointAngles);
        }
        else if(this.verbose){
            Debug.Log("Found no solution!");
        }
        return no_of_sols;
    }
    public int SetEEPose(Vector3 position, Quaternion orientation){ return SetEEPose(position, orientation, false);}

    private void SetArmCollision(bool enabled){
        Collider col;
        if(this.verbose){
            Debug.Log("Set collision of arm to " + enabled);
        }
        for(int i = 0; i < 7; i++){
            col = this.JointTransforms[i].GetComponent<Collider>();
            if(ignore_collisions)
                col.enabled = false;
            else
                col.enabled = enabled;
        }
    }

    private void SetArmHjsLimits(bool enabled){
        if(this.physics_enabled){
            if(!ignore_collisions){
                this.MJoints[1].SetLimit(enabled);                    
                this.MJoints[3].SetLimit(enabled);                    
                this.MJoints[5].SetLimit(enabled);                    
            }
            else{
                this.MJoints[1].SetLimit(false);                    
                this.MJoints[3].SetLimit(false);                    
                this.MJoints[5].SetLimit(false);                    
            }
        }
    }
    
    public void ResetPose(){
        SetArmCollision(false); 
        SetArmHjsLimits(false);       
        this.reset_pose = true;        
        SetEEPose(StartingPose.localPosition, StartingPose.localRotation, true);
    }

    void FixedUpdate(){        
        bool all_joints_set;
        all_joints_set = true;
        if(this.physics_enabled){
            for(int i = 0; i < 7; i++){
                all_joints_set = all_joints_set && this.MJoints[i].JointReachedTargetPosition();         
            }        
            this.joints_reached_target_position = all_joints_set;
            if(this.reset_pose && this.joints_reached_target_position){
                this.reset_pose = false;
                SetArmCollision(true);
                SetArmHjsLimits(true);
            }

            if(!all_joints_set && !this.joint_timed_out){
                this.time_out_counter++;
            }
            if(this.time_out_counter > MaxStepsBeforeTimeout){
                this.joint_timed_out = true;
                this.time_out_counter = 0;
            }
        }        
    }

    public bool JointsPositionSet(){return this.joints_reached_target_position;}

    public bool HasTimedOut(){return this.joint_timed_out;}
}