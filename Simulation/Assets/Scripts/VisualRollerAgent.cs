using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

using System;
// using UnityEngine;
using System.Linq;
// using Unity.MLAgents;
using UnityEngine.Serialization;

public class VisualRollerAgent : Agent{

    private Rigidbody rBody;
    public Transform Target, Enemy;
    public RegisterStringLogSideChannel comm_channel;
    public float curr_reward, dist_to_target,speed,calculated_r,on_hit_enemy;
    public int my_max_step_count, step_id;
    private bool parameters_set;
    public float on_hit_r, on_leave_arena_r, distance_multiplier, seed, agent_no, curr_episode, curr_step;
    public string agent_name;

    void Start (){
        rBody = GetComponent<Rigidbody>();
        this.parameters_set = false;
        this.on_hit_r = 1.0f;
        this.on_leave_arena_r = -1.0f;
        this.distance_multiplier = 10.0f;
        this.on_hit_enemy = 2.0f;
        this.speed = 12.5f;
        // this.speed = 25.0f;
        UnityEngine.Random.InitState(0);

        this.curr_episode = 0.0f;                
        this.curr_step = 0.0f;                
    }

    public override void OnEpisodeBegin(){
        // float plane_size = 20.0f - 1.0f;
        this.step_id = 0;
        this.curr_episode += 1.0f;                
        this.curr_step = 0.0f;       
        if(!this.parameters_set){
            var envParameters = Academy.Instance.EnvironmentParameters;
            if(envParameters.GetWithDefault("parameters_set", 0.0f) == 1.0f){
                this.seed = envParameters.GetWithDefault("seed", 0.0f);
                this.on_hit_r = envParameters.GetWithDefault("on_hit_target", 1.0f);
                this.on_hit_enemy = envParameters.GetWithDefault("on_hit_enemy", 0.1f);
                this.speed = envParameters.GetWithDefault("speed", 12.5f);
                this.on_leave_arena_r = envParameters.GetWithDefault("on_leave_arena", 0.0f);
                this.distance_multiplier = envParameters.GetWithDefault("distance_multiplier", 0.0f);
                this.parameters_set = true;

                UnityEngine.Random.InitState((int) this.seed);
            }
        }
        // if (this.transform.localPosition.y < 0){
            // If the Agent fell, zero its momentum
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3( 0, 0.5f, 0);
            // this.transform.localPosition = new Vector3( this.test_x, 0.5f, this.test_z);
            // this.test_x++;
            // this.test_z++;
        // }        
        // this.rBody.angularVelocity = Vector3.zero;
        // this.rBody.velocity = Vector3.zero;
        // this.transform.localPosition = new Vector3(UnityEngine.Random.value * plane_size - plane_size/2,
        //                                    0.5f,
        //                                    UnityEngine.Random.value * plane_size - plane_size/2);
        this.curr_reward = 0;
        // Move the target to a new spot
        this.Target.localPosition = new Vector3(-9.9f, 0.5f, -9.9f);        
        
        // float x_candidate, z_candidate;
        // x_candidate = UnityEngine.Random.value * plane_size - plane_size/2;
        // if(x_candidate < 0)
        //     x_candidate = Mathf.Min(x_candidate, -2.0f);
        // else
        //     x_candidate = Mathf.Max(x_candidate, 2.0f);

        // z_candidate = Mathf.Clamp(UnityEngine.Random.value * plane_size - plane_size/2, -2.0f, 2.0f);
        // if(z_candidate < 0)
        //     z_candidate = Mathf.Min(z_candidate, -2.0f);
        // else
        //     z_candidate = Mathf.Max(z_candidate, 2.0f);
        // this.Target.localPosition = new Vector3(x_candidate, 0.5f, z_candidate);

        // Move the enemy to a new spot
        // float lerpv = UnityEngine.Random.value;        
        // Vector3 clearance = (this.Target.localPosition - this.transform.localPosition).normalized * 1.5f;
        // this.Enemy.localPosition = Vector3.Lerp(this.transform.localPosition, this.Target.localPosition - clearance, lerpv);
        // Debug.Log("Episode Begin Agent" + agent_name +"!");
    }    

    public override void CollectObservations(VectorSensor sensor){

        // this.agent_camera.Render();
        // Target and Agent positions
        // Vector3 relative_direction = Target.localPosition - this.transform.localPosition;
        // relative_direction = relative_direction/18.0f;
        // sensor.AddObservation(relative_direction.x);
        // sensor.AddObservation(relative_direction.z);
        // tlocal = Target.localPosition;
        // alocal = this.transform.localPosition;
        sensor.AddObservation(Target.localPosition.x/10.0f);
        sensor.AddObservation(Target.localPosition.z/10.0f);
        // sensor.AddObservation(Enemy.localPosition.x/10.0f);
        // sensor.AddObservation(Enemy.localPosition.z/10.0f);
        sensor.AddObservation(this.transform.localPosition.x/10.0f);
        sensor.AddObservation(this.transform.localPosition.z/10.0f);
        // sensor.AddObservation(this.test_x);
        // sensor.AddObservation(this.test_z);
        // Debug.Log(rBody.velocity.x/17.0f);
        // Debug.Log(rBody.velocity.z/17.0f);
        // Agent velocity
        // sensor.AddObservation(rBody.velocity.x/17.0f);
        // sensor.AddObservation(rBody.velocity.z/17.0f);
        // Debug.Log(this.step_id++);
        // Debug.Log(this.transform.localPosition.x/10.0f);
        // Debug.Log(this.transform.localPosition.z/10.0f);
        // sensor.AddObservation(this.agent_no);
        // sensor.AddObservation(this.curr_step);
        // sensor.AddObservation(this.curr_episode);
        this.curr_step+=1.0f;
    }
    
    private float gauss(float amplitude, float spread, float x, float y, float x_center, float y_center){
        // Debug.Log((((x - x_center)*(x - x_center) + (y - y_center)*(y - y_center))/2*spread*spread));
        return amplitude * Mathf.Exp( - (((x - x_center)*(x - x_center) + (y - y_center)*(y - y_center))/(2*spread*spread)));
    }

    public float tw, ew, a, s, ao, so;
    public override void OnActionReceived(float[] vectorAction){        
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        // float calculated_r;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        // rBody.AddForce(controlSignal * speed);
        rBody.position = rBody.position + controlSignal * speed;
        // Debug.Log(rBody.position.x+", "+rBody.position.y+", "+rBody.position.z);        
        this.rBody.position = new Vector3(
            Mathf.Clamp(rBody.position.x, -10.0f, 10.0f),
            rBody.position.y,
            Mathf.Clamp(rBody.position.z, -10.0f, 10.0f)
        );

        // Rewards
        
        // tw = 0.9f;
        // ew = 0.1f;
        // a = 4f;
        // s = 10f;
        // ao = 4.0f;
        // so = .5f;
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition)/28.0f;
        dist_to_target = distanceToTarget;        
        // calculated_r = tw * this.gauss(a, s, this.transform.localPosition.x, this.transform.localPosition.z, Target.localPosition.x, Target.localPosition.z) - ew * this.gauss(ao, so, this.transform.localPosition.x, this.transform.localPosition.z, Enemy.localPosition.x, Enemy.localPosition.z) - a * tw;
        // calculated_r = this.gauss(a, s, this.transform.localPosition.x, this.transform.localPosition.z, Target.localPosition.x, Target.localPosition.z) - a;
        // calculated_r = this.distance_multiplier;
        calculated_r = -(distanceToTarget)*(distanceToTarget);
        // calculated_r = this.agent_no/10.0f;
        // calculated_r = -0.01f;
        // calculated_r /= this.my_max_step_count;
        // calculated_r = 1.0f/(distanceToTarget*9.0f*this.my_max_step_count);
        // calculated_r = -10.0f*Mathf.Log(distanceToTarget) + 25.0f;
        
        this.curr_reward += calculated_r;
        SetReward(calculated_r);
        var hit = Physics.OverlapBox(this.transform.position, new Vector3(1.00f, 1.0f, 1.00f));
        if (hit.Where(col => col.gameObject.CompareTag("target")).ToArray().Length == 1){
        //    Debug.Log(this.curr_reward);
            // SetReward(this.on_hit_r);
            // comm_channel.SendString("success");
            Debug.Log("Ended by touch: " + agent_no);
            EndEpisode();
        }
        
        // hit = Physics.OverlapBox(this.transform.position, new Vector3(1.00f, 1.0f, 1.00f));
        // if (hit.Where(col => col.gameObject.CompareTag("enemy")).ToArray().Length == 1){
        //     // Debug.Log("aaaa");
        //     SetReward(this.on_hit_enemy);
        //     // comm_channel.SendString("success");
        //     // EndEpisode();
        // }        
   
        // Fell off platform
        // if ((this.transform.localPosition.y < 0)){
        //     // Debug.Log(this.curr_reward);
        //     SetReward(this.on_leave_arena_r);
        //     // comm_channel.SendString("failure");
        //     // SetReward(-9.9f);
        //     // Debug.Log("Ended by falling: " + agent_no);
        //     EndEpisode();
        // }
        // if(this.StepCount > this.my_max_step_count){
        //     // SetReward(this.on_leave_arena_r*10.0f);
        //     // comm_channel.SendString("failure");
        //     EndEpisode();
        // }
        // float end_chance;
        // if(this.agent_no == 0)
        //     end_chance = 0.001f;                 
        // else
        //     end_chance = 0.005f;                 

        // if((this.curr_step > 16) && (UnityEngine.Random.value <= end_chance)){ // 5% of chance to end episode randomly
        //     // SetReward(this.on_leave_arena_r*10.0f);
        //     // comm_channel.SendString("failure");
        //     // Debug.Log("Ended randomly: " + agent_no);
        //     EndEpisode();
        // }
    }

    public override void Heuristic(float[] actionsOut){
        actionsOut[0] = Input.GetAxis("Horizontal");
        actionsOut[1] = Input.GetAxis("Vertical");
    }
}
