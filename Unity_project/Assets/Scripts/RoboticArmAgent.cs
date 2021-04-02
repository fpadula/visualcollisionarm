using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using System.Linq;

public class RoboticArmAgent : Agent{


    public bool debug_mode, ignore_sol_dist, ignore_collisions;
    public float[] JointAngles;

    public ArmController ac;
    public Transform Target, Obstacle, TargetSpawnSphere, EERefPos, AgentRefPos;

    public float min_spawn_radius, max_spawn_radius, target_distance;
    public float field_of_motion_angle, no_solution, max_steps, seed, on_hit_target, distance_to_end, on_timeout, on_hit_obstacle, on_hit_self, movespd, c1,c2,c3,c4, delta, dref, p;
    public Vector3 target_in_camera_coords, spawn_center;
    public Camera agent_camera;

    public float curr_steps;

    private bool parameters_set, training, collision_ends_episode, ee_only_collision, solid_obstacle;
    private Vector3 initial_agent_pos, initial_eeref_pos;
    private Quaternion initial_agent_rot;

    public void DrawVector(Vector3 starting_point, Vector3 dir, Color color){
        Debug.DrawLine(starting_point, starting_point + dir, color);
    }

    private float MapValue(float value, float from1, float to1, float from2, float to2) {
        return value*((to1 - to2)/(from1 - from2)) + (-to1*from2 + from1 * to2)/(from1 - from2);
    }

    // Start is called before the first frame update
    void Start(){
        this.parameters_set = false;

        this.initial_eeref_pos = this.transform.parent.InverseTransformPoint(this.AgentRefPos.position);
        this.initial_agent_pos = this.transform.localPosition;
        this.initial_agent_rot = this.transform.localRotation;

        if(this.ac != null){
            this.ac.ignore_collisions = this.ignore_collisions;
            this.ac.ignore_sol_dist = this.ignore_sol_dist;
        }
        this.training = false;
    }

    private void ResetAgentPosOri(){
        Rigidbody rb;
        rb = GetComponent<Rigidbody>();
        this.transform.localPosition = this.initial_agent_pos;
        this.transform.localRotation = this.initial_agent_rot;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }

    private void ReadParameters(){
        var envParameters = Academy.Instance.EnvironmentParameters;
        if(envParameters.GetWithDefault("parameters_set", 0.0f) == 1.0f){
            this.seed = envParameters.GetWithDefault("seed", 0.0f);
            this.on_hit_target = envParameters.GetWithDefault("on_hit_target", 1.0f);
            this.on_hit_obstacle = envParameters.GetWithDefault("on_hit_obstacle", -1.0f);
            this.on_hit_self = envParameters.GetWithDefault("on_hit_self", 1.0f);
            this.no_solution = envParameters.GetWithDefault("no_solution", -1.0f);
            this.movespd = envParameters.GetWithDefault("speed", 0.005f);
            this.field_of_motion_angle = envParameters.GetWithDefault("field_of_motion_angle", 5.0f);
            this.on_timeout = envParameters.GetWithDefault("on_timeout", -10.0f);
            this.c1 = envParameters.GetWithDefault("c1", 1.0f);
            this.c2 = envParameters.GetWithDefault("c2", 1.0f);
            this.c3 = envParameters.GetWithDefault("c3", 1.0f);
            this.c4 = envParameters.GetWithDefault("c4", 1.0f);
            this.delta = envParameters.GetWithDefault("delta", 1.0f);
            this.dref = envParameters.GetWithDefault("dref", 1.0f);
            this.p = envParameters.GetWithDefault("p", 1.0f);
            this.distance_to_end = envParameters.GetWithDefault("distance_to_end", 0.1f);
            this.max_steps = envParameters.GetWithDefault("max_steps", 500.0f);
            this.training = (envParameters.GetWithDefault("training", 0.0f) == 1.0f);
            this.collision_ends_episode = (envParameters.GetWithDefault("collision_ends_episode", 0.0f) == 1.0f);
            this.ee_only_collision = (envParameters.GetWithDefault("ee_only_collision", 0.0f) == 1.0f);
            this.Obstacle.GetComponent<Collider>().isTrigger = (envParameters.GetWithDefault("solid_obstacle", 1.0f) == 0.0f);
            // NO ARM TEST
            if(this.ac != null){
                this.ac.SetMaxJointSolDist( (double)envParameters.GetWithDefault("max_sol_joint_distance", 5.0f) );
                this.ac.SetMaxCartSolDist( (double)envParameters.GetWithDefault("max_sol_cart_distance", 5.0f) );
            }
            this.parameters_set = true;

            UnityEngine.Random.InitState((int) this.seed);
        }
    }
    // public float rtest,thetatest,phitest, otv, orv;

    public float min_agent_dist, min_targ_dist;
    public override void OnEpisodeBegin(){
        float theta, phi, radius, tx, ty, tz, ox, oy, object_theta, object_radius;
        Vector3 min_allowed_pos, max_allowed_pos,p0, u,v,n;
        // min_agent_dist = 0.1f;
        // min_targ_dist = 0.12f;

        ep_reward = 0.0f;
        this.curr_steps = 0;
        if(!this.parameters_set)
            ReadParameters();

        // Target.localPosition = this.ac.GetRandomValidPos();
        ResetAgentPosOri();
        // NO ARM TEST
        if(this.ac != null)
            this.ac.ResetPose();

        radius = MapValue(Random.value, 0, this.min_spawn_radius, 1.0f, this.max_spawn_radius);
        theta = MapValue(Random.value, 0, 0f, 1.0f, 100.0f)*Mathf.Deg2Rad;
        phi = MapValue(Random.value, 0, 0, 1.0f, 180.0f)*Mathf.Deg2Rad;
        // Move the target sphere to a new spot
        tx = radius * Mathf.Sin(theta) * Mathf.Cos(phi);
        ty = radius * Mathf.Sin(theta) * Mathf.Sin(phi);
        tz = radius * Mathf.Cos(theta);
        Target.localPosition = new Vector3(tx, ty, tz) + this.initial_eeref_pos;


        // radius = .25f;
        // theta = Random.value * Mathf.PI;
        // phi = Random.value * 2f * Mathf.PI;

        // // Random target position inside spawn sphere
        // tx = radius * Mathf.Sin(theta) * Mathf.Cos(phi);
        // ty = radius * Mathf.Sin(theta) * Mathf.Sin(phi);
        // tz = radius * Mathf.Cos(theta);
        // Target.localPosition = new Vector3(tx, ty, tz) + TargetSpawnSphere.localPosition;

        min_allowed_pos = this.initial_eeref_pos + min_agent_dist*(Target.localPosition - this.initial_eeref_pos).normalized;
        max_allowed_pos = Target.localPosition - min_targ_dist*(Target.localPosition - this.initial_eeref_pos).normalized;
        p0 = min_allowed_pos + (max_allowed_pos - min_allowed_pos)* Random.value;
        n = (Target.localPosition - this.initial_eeref_pos).normalized;
        // Debug.Log(Vector3.Distance(p0, this.initial_agent_pos));
        // U and V need to be orthogonal to n.
        if(( !(Mathf.Abs(n.z) < 1e-5) || !(Mathf.Abs(n.x) < 1e-5)) )
            v = new Vector3(-n.z, 0, n.x).normalized;
        else
            v = new Vector3(-n.y, n.x, 0).normalized;

        u = Vector3.Cross(n, v).normalized;

        object_theta = MapValue(Random.value, 0, 0, 1.0f, 360.0f)*Mathf.Deg2Rad;
        object_radius = MapValue(Random.value, 0, 0, 1.0f, .3f);
        ox = object_radius * Mathf.Cos(object_theta);
        oy = object_radius * Mathf.Sin(object_theta);
        this.Obstacle.localPosition = p0 + ox*v + oy*u;
        // this.Obstacle.position = p0;
    }


    private bool is_on_camera(Vector3 coord){
        return ((coord.z >= 0) && (coord.x >= 0.0f) && (coord.x <= 1.0f) && (coord.y >= 0.0f) && (coord.y <= 1.0f));
    }

    public override void CollectObservations(VectorSensor sensor){
        float x,y,z, min_z;

        // Adding the z-depth of the object that is closest to the camera, if any is present.
        // min_z = 99.0f;
        // // target_in_camera_coords = agent_camera.WorldToViewportPoint(this.Target.position);
        // // if(is_on_camera(target_in_camera_coords)){ // Inside X and Y
        // //     min_z = target_in_camera_coords.z;
        // // }
        // z = -1.0f;
        // y = -1.0f;
        // x = -1.0f;
        // target_in_camera_coords = agent_camera.WorldToViewportPoint(this.Obstacle.position);
        // if(is_on_camera(target_in_camera_coords)){ // Inside X and Y
        //     if(target_in_camera_coords.z < min_z){
        //         min_z = target_in_camera_coords.z;
        //         x = target_in_camera_coords.x;
        //         y = target_in_camera_coords.y;
        //     }
        // }
        // // target_in_camera_coords = agent_camera.WorldToViewportPoint(ac.Base.position);
        // // if(is_on_camera(target_in_camera_coords)){ // Inside X and Y
        // //     if(target_in_camera_coords.z < min_z){
        // //         min_z = target_in_camera_coords.z;
        // //         x = target_in_camera_coords.x;
        // //         y = target_in_camera_coords.y;
        // //     }
        // // }
        // // for(int i = 0; i < 5; i++){
        // //     target_in_camera_coords = agent_camera.WorldToViewportPoint(ac.GetJointPos(i, true));
        // //     if(is_on_camera(target_in_camera_coords)){ // Inside X and Y
        // //         if(target_in_camera_coords.z < min_z){
        // //             min_z = target_in_camera_coords.z;
        // //             x = target_in_camera_coords.x;
        // //             y = target_in_camera_coords.y;
        // //         }
        // //     }
        // // }
        // if(min_z != 99.0f)
        //     z = min_z;
        // else{
        //     z = 10.0f;
        //     y = 0.0f;
        //     x = 0.0f;
        // }
        sensor.AddObservation(Obstacle.localPosition.x);
        sensor.AddObservation(Obstacle.localPosition.y-1);
        sensor.AddObservation(Obstacle.localPosition.z);
        // Debug.Log("X: " + x + ", Y: " + y + ", Z:" + z);
        // sensor.AddObservation(x);
        // sensor.AddObservation(y);
        // sensor.AddObservation(z);

        // Goal info (either spawn sphere or target)
        // sensor.AddObservation(TargetSpawnSphere.localPosition.x);
        // sensor.AddObservation(TargetSpawnSphere.localPosition.y);
        // sensor.AddObservation(TargetSpawnSphere.localPosition.z);
        sensor.AddObservation(Target.localPosition.x);
        sensor.AddObservation(Target.localPosition.y-1);
        sensor.AddObservation(Target.localPosition.z);

        // Planner info
        // Debug.Log(transform.localPosition);
        sensor.AddObservation(transform.localPosition.x);
        sensor.AddObservation(transform.localPosition.y-1);
        sensor.AddObservation(transform.localPosition.z);
        sensor.AddObservation(transform.localRotation);

        // Obstacle info (remove if using vision)
        // sensor.AddObservation(Obstacle.localPosition);


        // NO ARM TEST
        if(this.ac != null){
            // Arm info:
            sensor.AddObservation(this.ac.JointAngles[0]/180.0f);
            sensor.AddObservation(this.ac.JointAngles[1]/180.0f);
            sensor.AddObservation(this.ac.JointAngles[3]/180.0f);
            sensor.AddObservation(this.ac.JointAngles[4]/180.0f);
            sensor.AddObservation(this.ac.JointAngles[5]/180.0f);
            sensor.AddObservation(this.ac.JointAngles[6]/180.0f);

            // sensor.AddObservation(ac.GetJointPos(3).x);
            // sensor.AddObservation(ac.GetJointPos(3).y-1);
            // sensor.AddObservation(ac.GetJointPos(3).z);

            // sensor.AddObservation(ac.GetJointPos(5).x);
            // sensor.AddObservation(ac.GetJointPos(5).y-1);
            // sensor.AddObservation(ac.GetJointPos(5).z);

            sensor.AddObservation(ac.GetJointPos(6).x);
            sensor.AddObservation(ac.GetJointPos(6).y-1);
            sensor.AddObservation(ac.GetJointPos(6).z);

            sensor.AddObservation(ac.GetJointRot(6, true));
        }
    }

    // Update is called once per frame
    public override void Heuristic(float[] actionsOut){
        actionsOut[0] = 0.5f*Input.GetAxis("Horizontal");
        actionsOut[1] = 0.5f*Input.GetAxis("Vertical");
        actionsOut[2] = 0.5f*Input.GetAxis("Speed");
        // actionsOut[3] = 0.5f*Input.GetAxis("Speed");
        // actionsOut[4] = 0.5f*Input.GetAxis("Speed");
        // h_axis = actionsOut[0];
        // v_axis = actionsOut[1];
    }

    private float GaussianF(float amplitude, float spread, Vector3 center, Vector3 coord){
        float numerator = (coord.x - center.x)*(coord.x - center.x) + (coord.y - center.y)*(coord.y - center.y) + (coord.z - center.z)*(coord.z - center.z);
        return amplitude * Mathf.Exp( - (numerator/(2 * spread * spread)));
    }

    public float goal_w, obst_w, goal_amp, goal_spread, obst_amp, obst_spread;
    private float GaussianReward(Vector3 obs_p, Vector3 goal_p, Vector3 agent_p){
        // goal_w = 0.8f;
        // obst_w = 0.2f;
        // goal_amp = 4.0f;
        // goal_spread = 10.0f;
        // obst_amp = 4.0f;
        // obst_spread = 2.0f;

        // return goal_w *(GaussianF(goal_amp, goal_spread, goal_p, agent_p) - goal_amp);
        return goal_w * (GaussianF(goal_amp, goal_spread, goal_p, agent_p) - goal_amp) - obst_w * GaussianF(obst_amp, obst_spread, obs_p, agent_p);
    }

    private float HuberLoss(float a, float delta){
        if(Mathf.Abs(a) <= delta){
            return (a*a)/2.0f;
        }
        else{
            return delta*(Mathf.Abs(a) - delta/2.0f);
        }
    }

    public float ep_reward, sphereRadius, look_at_angle, decoupling_distance, planner_distance;
    public Vector3 col_offset;

    public override void OnActionReceived(float[] vectorAction){
        float step_reward, object_distance;
        Vector3 new_position, a_to_t;
        Quaternion movement_rotation;
        int no_sols, layerMask;
        //  The action for this agent consists of moving in a cone
        //      vectorAction[0]: rotation around y-axis
        //      vectorAction[1]: rotation around x-axis
        //      vectorAction[2]: movement magnitude
        // Debug.Log(vectorAction[0] + ", " + vectorAction[1] + ", " + vectorAction[2]);
        movement_rotation = Quaternion.Euler(
            vectorAction[0]*field_of_motion_angle,
            vectorAction[1]*field_of_motion_angle,
            vectorAction[2]*field_of_motion_angle);
        // movement_rotation = Quaternion.Euler(
        //     -vectorAction[1]*field_of_motion_angle,
        //     vectorAction[0]*field_of_motion_angle,
        //     0);

        new_position = this.transform.localPosition + new Vector3(vectorAction[3], vectorAction[4], vectorAction[5]) * this.movespd;
        // new_position = this.transform.localPosition + this.transform.forward * this.movespd * vectorAction[2];
        new_position.z = Mathf.Clamp(new_position.z, -1.0f, 1.0f);
        new_position.x = Mathf.Clamp(new_position.x, -1.0f, 1.0f);
        new_position.y = Mathf.Clamp(new_position.y, 0.0f, 2.0f);

        movement_rotation = this.transform.localRotation * movement_rotation;

        // if(Vector3.Distance(ac.GetJointPos(6), transform.position) >= 0.05f){
        //     no_of_big_jumps = 5;
        // }
        // NO ARM TEST
        // no_sols = 1;
        if(this.ac != null){
            if(this.ac.HasTimedOut()){
                if(debug_mode)
                    Debug.Log("Arm was not able to reach target joint angles! Ending episode...");
                SetReward(this.on_timeout);
                EndEpisode();
                return;
            }
            no_sols = this.ac.SetEEPose(new_position, movement_rotation);
        }
        else{
            no_sols = 1;
        }
        this.transform.localPosition = new_position;
        this.transform.localRotation = movement_rotation;


        // Rewards
        // NO ARM TEST
        if(this.ac != null){
            // this.target_distance = Vector3.Distance(ac.GetJointPos(6, true), Target.position);
            decoupling_distance = Vector3.Distance(ac.GetJointPos(6, true), this.transform.position);
            if(decoupling_distance <= 0.02f)
                decoupling_distance = 0;
        }
        else{
            decoupling_distance = 0;
        }
        this.target_distance = Vector3.Distance(this.EERefPos.position, Target.position);
        planner_distance = Vector3.Distance(Target.position, this.transform.position);
        object_distance = Vector3.Distance(Obstacle.position, this.EERefPos.position);
        // Debug.Log(object_distance);
        // Normalizing distances:
        // this.target_distance /= 3.4641f;
        // planner_distance /= 3.4641f;
        // object_distance /= 3.4641f;
        // float delta = 0.5f;
        // planner_distance = HuberLoss(planner_distance, delta);
        // object_distance = HuberLoss(object_distance, delta);

        a_to_t = Target.position - this.transform.position;
        look_at_angle = Vector3.SignedAngle(this.transform.forward, a_to_t, this.transform.right)*Mathf.Deg2Rad;
        // action_diff = new Vector3(
        //     vectorAction[0] - this.prev_action[0],
        //     vectorAction[1] - this.prev_action[1],
        //     vectorAction[2] - this.prev_action[2]
        //     );
        // Debug.Log(- 5.0f/(1.0f + 5.0f*Vector3.Distance(Obstacle.position, this.transform.position)));
        // step_reward = -this.target_distance - decoupling_distance - this.on_hit_obstacle/(1.0f - object_distance);
        // step_reward = -this.target_distance - decoupling_distance + 0.5f*(- 1 + Mathf.Cos(look_at_angle));
        // step_reward = -this.target_distance;        
        step_reward = -this.c1*HuberLoss(this.target_distance, delta) - this.c2*Mathf.Pow((this.dref/(object_distance+this.dref)), this.p) - this.c3*(1 - Mathf.Cos(look_at_angle)) - this.c4*decoupling_distance;
        // step_reward = -this.target_distance - 1 + Mathf.Cos(look_at_angle) - 5.0f/(1.0f + Vector3.Distance(Obstacle.position, this.transform.position));
        // step_reward = GaussianReward(Obstacle.position, Target.position, transform.position);
        // Debug.Log(step_reward*this.reward_mag_scalling + ", obj: " + Mathf.Pow((0.2f/(object_distance+0.2f)), 8.0f) + ", target: " + this.target_distance);
        AddReward(step_reward);
        // step_reward = -(this.target_distance);
        // step_reward = -1 - (this.target_distance)*(this.target_distance) + Mathf.Cos(look_at_angle) - action_diff.magnitude;
        // step_retransform.positionward /= this.MaxStep;
        ep_reward += step_reward;
        if(no_sols == 0){
            if(debug_mode)
                Debug.Log("No solution for this agent pose! (" + this.no_solution + ")");
            AddReward(this.no_solution);
        }

        // Checking collision against target
        // Debug.Log(Vector3.Distance(this.transform.position, Target.position));
        if (this.target_distance < distance_to_end){
            if(debug_mode)
                Debug.Log("Hit target!");
            SetReward(this.on_hit_target);
            EndEpisode();
            return;
        }

        // // Checking collision against arm
        // layerMask = 1 << 11;
        // if (Physics.CheckSphere(
        //         ac.GetJointPos(6, true) + ac.GetJointRot(6, true)*col_offset,
        //         sphereRadius,
        //         layerMask
        //         )){
        //     if(debug_mode)
        //         Debug.Log("Collision of EE against arm! (" + this.on_hit_self + ")");
        //     AddReward(this.on_hit_self);
        // }

        // Checking collision of the arm against obstacles
        if(ee_only_collision)
            layerMask = 1 << 8;
        else
            layerMask = 1 << 11 | 1 << 8;
        // // if (Physics.CheckBox(this.Obstacle.position,this.Obstacle.lossyScale/2.0f,this.Obstacle.rotation,layerMask)){
        if ( (this.Obstacle.gameObject.activeSelf) && (Physics.CheckSphere(this.Obstacle.position,obstacle_radius,layerMask))){
        // // if (Physics.CheckSphere(this.Obstacle.position,this.Obstacle.lossyScale.x,layerMask)){
        // // if (Physics.CheckBox(Target.position,Target.lossyScale/2.0f,Target.rotation,layerMask)){
            if(debug_mode)
                Debug.Log("Collision of Arm againt obstacle! (" + this.on_hit_obstacle + ")");
            AddReward(this.on_hit_obstacle);
            if(this.collision_ends_episode)
                EndEpisode();
        }

        if(this.curr_steps >= this.max_steps){
            // Debug.Log("Max number of steps reached!");
            EndEpisode();
            return;
        }
        this.curr_steps++;
    }
    public Vector3 debugcube;
    public float obstacle_radius;
    void OnDrawGizmosSelected(){
        if(debug_mode){
            // Draw a yellow sphere at the transform's position
            Gizmos.color = Color.yellow;
            if(this.ac != null)
                Gizmos.DrawSphere(ac.GetJointPos(6,true)+ac.GetJointRot(6,true)*col_offset, sphereRadius);
            Gizmos.DrawSphere(this.Obstacle.position, obstacle_radius);
            Gizmos.DrawSphere(this.Target.position, obstacle_radius);
        }
        
    }

    public int decision_req_interval = 5;
    private int decision_req_counter = 0;
    private void FixedUpdate(){                
        // NO ARM TEST
        // if(this.ac.HasTimedOut())
        //     this.ac.ResetPose();
        if(this.ac != null){
            if(((decision_req_counter <= 0) || this.training) && (this.ac.JointsPositionSet() || this.ac.HasTimedOut())){
                RequestDecision();
                decision_req_counter = decision_req_interval;
            }
            else
                decision_req_counter--;
        }
        else{
            if((decision_req_counter <= 0) || this.training){
                RequestDecision();
                decision_req_counter = decision_req_interval;
            }
            else
                decision_req_counter--;
        }

        //     // this.ac.ResetPose();
        // if (this.ac.JointsPositionSet()) || this.ac.HasTimedOut())
        //     RequestDecision();

        // if (this.ac.JointsPositionSet())
            // RequestDecision();
    }

}

