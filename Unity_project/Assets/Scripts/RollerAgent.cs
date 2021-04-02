using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class RollerAgent : Agent{
    
    private Rigidbody rBody;
    public Transform Target;
    public RegisterStringLogSideChannel comm_channel;
    public float curr_reward;
    private bool parameters_set;
    private float on_hit_r, on_leave_arena_r, distance_multiplier;

    void Start (){
        rBody = GetComponent<Rigidbody>();
        this.parameters_set = false;    
        this.on_hit_r = 1.0f;
        this.on_leave_arena_r = -1.0f;
        this.distance_multiplier = 1.0f;
    }

    public override void OnEpisodeBegin(){
        if(!this.parameters_set){
            var envParameters = Academy.Instance.EnvironmentParameters;
            if(envParameters.GetWithDefault("parameters_set", 0.0f) == 1.0f){
                this.on_hit_r = envParameters.GetWithDefault("on_hit_target", 0.0f);
                this.on_leave_arena_r = envParameters.GetWithDefault("on_leave_arena", 0.0f);
                this.distance_multiplier = envParameters.GetWithDefault("distance_multiplier", 0.0f);
                this.parameters_set = true;
            }
        }        
        if (this.transform.localPosition.y < 0){
            // If the Agent fell, zero its momentum
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3( 0, 0.5f, 0);
        }
        this.curr_reward = 0;
        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 8 - 4,
                                           0.5f,
                                           Random.value * 8 - 4);
    }

    public override void CollectObservations(VectorSensor sensor){
        

        // Target and Agent positions
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float speed = 10;
    public override void OnActionReceived(float[] vectorAction){        
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        rBody.AddForce(controlSignal * speed);                

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);        
        SetReward(-distanceToTarget*this.distance_multiplier);
        this.curr_reward += -distanceToTarget*this.distance_multiplier;
        // Reached target
        if (distanceToTarget < 1.42f){
            // Debug.Log(this.curr_reward);
            SetReward(this.on_hit_r);
            comm_channel.SendString("success");
            EndEpisode();
        }

        // Fell off platform
        if (this.transform.localPosition.y < 0){
            // Debug.Log(this.curr_reward);
            AddReward(this.on_leave_arena_r);
            comm_channel.SendString("failure");
            EndEpisode();
        }
        if(this.StepCount > 1000){ 
            comm_channel.SendString("failure");
            EndEpisode();
        }
    }

    public override void Heuristic(float[] actionsOut){
        actionsOut[0] = Input.GetAxis("Horizontal");
        actionsOut[1] = Input.GetAxis("Vertical");
    }
}
