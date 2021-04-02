using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

public class InverseKinematics : MonoBehaviour{

    [DllImport("ik")]
    private static extern int UnityIk(
        double[] t,
        double[] r,
        double[] JointBuffer,
        bool allow_big_jumps,
        int max_sol_tries,
        double max_joint_dist,
        double max_cart_dist,
        double cart_tol,
        double joint_tol,
        bool joint_limit_check
    );    

    [DllImport("ik")]
    private static extern void ComputeFk(
        double[] JointBuffer,
        double[] t,
        double[] r
    );    

    private double[] t;
    private double[] r;
    private double[] JointBuffer;    

    private Matrix4x4 Unity2RosHT, Ros2UnityHT;

    public int max_sol_tries;
    public double max_joint_dist, max_cart_dist, cart_tol, joint_tol;
    public bool joint_limit_check;

    

    // Start is called before the first frame update
    void Start(){
        this.t = new double[3];
        this.r = new double[9];
        this.JointBuffer = new double[7];        

        this.Unity2RosHT = Matrix4x4.zero;
        this.Unity2RosHT[0, 0] = -1.0f;
        this.Unity2RosHT[1, 2] = -1.0f;
        this.Unity2RosHT[2, 1] =  1.0f;        

        this.Ros2UnityHT = Matrix4x4.zero;
        this.Ros2UnityHT[0, 0] = -1.0f;
        this.Ros2UnityHT[1, 2] =  1.0f;
        this.Ros2UnityHT[2, 1] = -1.0f;  
    }

    public void UpdateJointBuffer(float[] joint_angles, bool angles_in_rad){
        for(int i = 0; i < 7; i++)
            this.JointBuffer[i] = angles_in_rad? (double)joint_angles[i] : (double)(Mathf.Deg2Rad*joint_angles[i]);
    }
    public void UpdateJointBuffer(float[] joint_angles){ UpdateJointBuffer(joint_angles, false);}

    public void UpdateJointAngles(float[] joint_angles, bool target_angles_in_rad){
        for(int i = 0; i < 7; i++)
            joint_angles[i] = target_angles_in_rad? (float)JointBuffer[i] : (float)(Mathf.Rad2Deg*this.JointBuffer[i]);
    }
    public void UpdateJointAngles(float[] joint_angles){ UpdateJointBuffer(joint_angles, false);}    

    public Vector3 ComputeFk(float[] joint_angles, bool in_rad){
        Vector3 ret;
        UpdateJointBuffer(joint_angles, in_rad);
        ComputeFk(this.JointBuffer,this.t,this.r);
        ret = new Vector3((float)this.t[0], (float)this.t[1], (float)this.t[2]);
        return Ros2UnityHT.MultiplyPoint3x4(ret);
        // return ret;
    }

    public int ComputeIK(Vector3 pos, Quaternion ori, bool ignore_sol_dist){
        Matrix4x4 unityHT, rosHT;
        Quaternion corrected_ori;

        // Performs a 90° rotation around X so that the EE faces the forward axis
        corrected_ori = new Quaternion(ori.x, ori.z, -ori.y, ori.w) * Quaternion.Euler(90.0f, 0, 0) ;
        unityHT = Matrix4x4.TRS(pos, corrected_ori, Vector3.one);
        rosHT = this.Unity2RosHT * unityHT;

        this.t[0] = rosHT[0,3];
        this.t[1] = rosHT[1,3];
        this.t[2] = rosHT[2,3];

        this.r[0] = unityHT[0,0];
        this.r[1] = unityHT[0,1];
        this.r[2] = unityHT[0,2];
        this.r[3] = unityHT[1,0];
        this.r[4] = unityHT[1,1];
        this.r[5] = unityHT[1,2];
        this.r[6] = unityHT[2,0];
        this.r[7] = unityHT[2,1];
        this.r[8] = unityHT[2,2];

        return UnityIk(t, r, JointBuffer, ignore_sol_dist, this.max_sol_tries, this.max_joint_dist, this.max_cart_dist, this.cart_tol, this.joint_tol, this.joint_limit_check);
    }
}
