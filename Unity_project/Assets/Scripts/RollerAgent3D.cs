using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class RollerAgent3D : Agent{
    
    private Rigidbody rBody;
    public Transform Target;    
    public float distancefromcenter, distanceToTarget;

    void Start (){
        rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin(){        
        if (Vector3.Distance(Vector3.zero, this.transform.localPosition) >= 0.4875f){
            // Agent is outside spawn sphere
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3( 0, 0, 0);
        }

        float theta, phi, radius;
        radius = 0.2f + Random.value * (0.4875f - 0.2f);
        theta = Random.value * Mathf.PI;
        phi = Random.value * 2f * Mathf.PI;

        // Move the target to a new spot
        // Target.localPosition = new Vector3(Random.value * 8 - 4,
        //                                    0.5f,
        //                                    Random.value * 8 - 4);
        Target.localPosition = new Vector3(radius * Mathf.Sin(theta) * Mathf.Cos(phi),
                                        radius * Mathf.Sin(theta) * Mathf.Sin(phi),
                                        radius * Mathf.Cos(theta));
    }

    public override void CollectObservations(VectorSensor sensor){
        // Target and Agent positions
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        sensor.AddObservation(rBody.velocity);        
    }

    public float speed = 10;
    public override void OnActionReceived(float[] vectorAction){
        // Actions, size = 3
        Vector3 controlSignal = new Vector3(vectorAction[0], vectorAction[1], vectorAction[2]);        
        rBody.AddForce(controlSignal * speed);

        // Rewards
        this.distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        SetReward(-distanceToTarget);
        // Reached target
        if (distanceToTarget < 0.15f){
            SetReward(10.0f);
            EndEpisode();
        }

        // Fell off platform
        if (Vector3.Distance(Vector3.zero, this.transform.localPosition) >= 0.4875f){
            AddReward(-10.0f);
            EndEpisode();
        }
    }

    public override void Heuristic(float[] actionsOut){
        int direction = Input.GetKeyDown(KeyCode.I)? 1 : (Input.GetKeyDown(KeyCode.K)? -1 : 0);        
        actionsOut[0] = Input.GetAxis("Horizontal");
        actionsOut[1] = Input.GetAxis("Vertical");
        actionsOut[2] = 0.1f * direction;
    }
}
