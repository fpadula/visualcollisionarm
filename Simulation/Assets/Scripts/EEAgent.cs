using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

using System.Linq;

public class EEAgent : Agent{

    public Transform Target, Obstacle;
    public float plane_size, field_of_motion_angle, dist_to_trigger, target_distance, seed, on_hit_r, on_leave_arena_r, on_hit_obstacle, movespd;
    private Vector3 min_coords, max_coords;
    private bool parameters_set;
    private Rigidbody rBody;
    // Debug variables:
    public Vector3 target_in_camera_coords;
    public Camera agent_camera;
    private float[] prev_action;

    public float mapvalue(float value, float from1, float to1, float from2, float to2) {
        return value*((to1 - to2)/(from1 - from2)) + (-to1*from2 + from1 * to2)/(from1 - from2);
    }

    public void DrawVector(Vector3 starting_point, Vector3 dir, Color color){
        Debug.DrawLine(starting_point, starting_point + dir, color);
    }

    // Start is called before the first frame update
    void Start(){
        rBody = GetComponent<Rigidbody>();
        this.parameters_set = false;
        this.min_coords = new Vector3(-plane_size/2.0f, 0, -plane_size/2.0f);
        this.max_coords = new Vector3(plane_size/2.0f, plane_size, plane_size/2.0f);
        this.min_coords = this.transform.parent.TransformPoint(this.min_coords);
        this.max_coords = this.transform.parent.TransformPoint(this.max_coords);
        this.prev_action = new float[3];
    }

    public override void OnEpisodeBegin(){
        float theta, phi, radius, tx, ty, tz;

        ep_reward = 0.0f;

        if(!this.parameters_set){
            var envParameters = Academy.Instance.EnvironmentParameters;
            if(envParameters.GetWithDefault("parameters_set", 0.0f) == 1.0f){
                this.seed = envParameters.GetWithDefault("seed", 0.0f);
                this.on_hit_r = envParameters.GetWithDefault("on_hit_target", 1.0f);
                this.on_leave_arena_r = envParameters.GetWithDefault("on_leave_arena", 0.0f);
                this.on_hit_obstacle = envParameters.GetWithDefault("on_hit_enemy", 0.1f);
                this.movespd = envParameters.GetWithDefault("speed", 12.5f);
                this.field_of_motion_angle = envParameters.GetWithDefault("field_of_motion_angle", 1.25f);
                this.parameters_set = true;

                UnityEngine.Random.InitState((int) this.seed);
            }
        }
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
        // this.transform.localPosition = new Vector3( 0, 0.5f, 0);
        // this.transform.localEulerAngles = new Vector3( 0, 0, 0);

        radius = Random.value * this.plane_size/2.0f;
        theta = Random.value * Mathf.PI;
        phi = Random.value * 2f * Mathf.PI;

        // Move the target to a new spot
        tx = radius * Mathf.Sin(theta) * Mathf.Cos(phi);
        ty = 0.51f + radius * Mathf.Sin(theta) * Mathf.Sin(phi);
        tz = radius * Mathf.Cos(theta);
        // Making sure the target is not bellow ground
        ty = Mathf.Max(ty, 0.060f);
        Target.localPosition = new Vector3(tx, ty, tz);

        // Setting enemy position along a line between the target and the agent
        Vector3 direction = (Target.localPosition - this.transform.localPosition).normalized;
        Vector3 endpoint, startpoint;
        endpoint = Target.localPosition - direction * .115f;
        startpoint = this.transform.localPosition + direction * .155f;
        this.Obstacle.localPosition = Vector3.Lerp(startpoint, endpoint, Random.value);
    }

    private bool is_on_camera(Vector3 coord){
        return ((coord.x >= 0.1f) && (coord.x <= 0.9f) && (coord.y >= 0.1f) && (coord.y <= 0.9f));
    }

    public override void CollectObservations(VectorSensor sensor){
        // target_in_camera_coords = camera.WorldToViewportPoint(this.Target.position);
        // target_in_camera_coords = camera.WorldToViewportPoint(this.Obstacle.position);

        // if(is_on_camera(target_in_camera_coords)){ // Inside X and Y
        //     sensor.AddObservation(target_in_camera_coords);
        //     // sensor.AddObservation(1.0f);
        // }
        // else{
        //     // sensor.AddObservation(0.0f);
        //     sensor.AddObservation(new Vector3(-1.0f, -1.0f, -1.0f));
        // }
        // Debug.Log(target_in_camera_coords);
        // Target and Agent positions
        // Debug.Log(this.transform.localPosition);
        sensor.AddObservation(Target.localPosition.x);
        sensor.AddObservation(Target.localPosition.y - 1.0f);
        sensor.AddObservation(Target.localPosition.z);
        sensor.AddObservation(this.transform.localPosition.x);
        sensor.AddObservation(this.transform.localPosition.y - 1.0f);
        sensor.AddObservation(this.transform.localPosition.z);
        // sensor.AddObservation(this.Obstacle.localPosition.x);
        // sensor.AddObservation(this.Obstacle.localPosition.y - 1.0f);
        // sensor.AddObservation(this.Obstacle.localPosition.z);


        // Vector3 a_to_t = Target.position - this.transform.position;        
        // sensor.AddObservation(Vector3.SignedAngle(this.transform.forward, a_to_t, this.transform.right)/180.0f);
        sensor.AddObservation(this.rBody.rotation);
    }

    // Update is called once per frame
    public override void Heuristic(float[] actionsOut){
        actionsOut[0] = Input.GetAxis("Horizontal");
        actionsOut[1] = Input.GetAxis("Vertical");
        actionsOut[2] = Input.GetAxis("Speed");
        // h_axis = actionsOut[0];
        // v_axis = actionsOut[1];
    }

    public float ep_reward, sphereRadius;
    public Vector3 col_offset;
    public override void OnActionReceived(float[] vectorAction){
        //  The action for this agent consists of moving in a cone
        //      vectorAction[0]: rotation around y-axis
        //      vectorAction[1]: rotation around x-axis
        //      vectorAction[2]: movement magnitude
        // Debug.Log(vectorAction[0] + ", " + vectorAction[1] + ", " + vectorAction[2]);

        float step_reward, look_at_angle;
        Vector3 new_position, a_to_t, action_diff;
        Quaternion movement_rotation;

        movement_rotation = Quaternion.Euler(-vectorAction[1]*field_of_motion_angle, vectorAction[0]*field_of_motion_angle, 0);

        new_position = this.rBody.position + this.transform.forward * this.movespd * vectorAction[2];

        new_position = new Vector3(
            Mathf.Clamp(new_position.x, this.min_coords.x, this.max_coords.x),
            Mathf.Clamp(new_position.y, this.min_coords.y, this.max_coords.y),
            Mathf.Clamp(new_position.z, this.min_coords.z, this.max_coords.z)
        );
        // this.transform.LookAt(new_position);
        this.rBody.rotation = this.rBody.rotation * movement_rotation;
        this.rBody.position = new_position;
        // Rewards
        this.target_distance = Vector3.Distance(this.transform.localPosition, Target.localPosition);
        // Normalizing distance:
        this.target_distance /= 3.4641f;

        a_to_t = Target.position - this.transform.position;
        look_at_angle = Vector3.SignedAngle(this.transform.forward, a_to_t, this.transform.right)*Mathf.Deg2Rad;
        action_diff = new Vector3(
            vectorAction[0] - this.prev_action[0],
            vectorAction[1] - this.prev_action[1],
            vectorAction[2] - this.prev_action[2]
            );
        // step_reward = -this.target_distance -1 + Mathf.Cos(look_at_angle);
        step_reward = -1 + Mathf.Cos(look_at_angle) - (this.target_distance);
        // step_reward = -1 - (this.target_distance)*(this.target_distance) + Mathf.Cos(look_at_angle) - action_diff.magnitude;
        // step_reward /= this.MaxStep;
        ep_reward += step_reward;
        SetReward(step_reward);

        // var hit = Physics.OverlapBox(this.rBody.position, new Vector3(0.091f, 0.112174f, 0.061925f)*2.0f, this.rBody.rotation);
        // if (hit.Where(col => col.gameObject.CompareTag("target")).ToArray().Length == 1){
        //     // SetReward(this.on_hit_r);
        //     EndEpisode();
        // }
        int layerMask = 1 << 9;
        if (Physics.CheckSphere(this.transform.position+col_offset, sphereRadius, layerMask)){
        // if (this.target_distance < dist_to_trigger){
            // SetReward(this.on_hit_r);
            EndEpisode();
        }

        
        layerMask = 1 << 11;
        if (Physics.CheckSphere(this.transform.position+col_offset, sphereRadius, layerMask)){
        // if (Vector3.Distance(this.transform.localPosition, Obstacle.localPosition) < 0.1f){
            SetReward(this.on_hit_obstacle);
            // Debug.Log("hit!!");
            // ep_reward += this.on_hit_obstacle;
            // EndEpisode();
        }
        this.prev_action[0] = vectorAction[0];
        this.prev_action[1] = vectorAction[1];
        this.prev_action[2] = vectorAction[2];
    }
    // void OnDrawGizmosSelected()
    // {
    //     // Draw a yellow sphere at the transform's position
    //     Gizmos.color = Color.yellow;
    //     Gizmos.DrawSphere(this.transform.position + col_offset, sphereRadius);
    // }
}
