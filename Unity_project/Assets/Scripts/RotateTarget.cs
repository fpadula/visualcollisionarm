using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateTarget : MonoBehaviour{
    public float spd, radius, signal;
    public int steps_to_change_signal;
    private int steps;
    public float alpha;
    // Start is called before the first frame update
    void Start(){
        this.alpha = Random.value * Mathf.PI * 2.0f;
        this.steps = 0;
    }

    // Update is called once per frame
    void Update(){
        Vector3 new_pos;
        new_pos = this.transform.localPosition;
        new_pos.x = this.radius * Mathf.Cos(signal*alpha);
        new_pos.z = this.radius * Mathf.Sin(signal*alpha);
        this.transform.localPosition = new_pos;
        this.alpha += spd * Time.deltaTime;
        if(this.alpha > Mathf.PI * 2.0f){
            this.alpha = 0.0f;
        }
        if(steps >= steps_to_change_signal){
            steps = 0;
            signal *= -1;
        }
        else{
            steps++;
        }
    }
}
