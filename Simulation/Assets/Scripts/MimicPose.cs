using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MimicPose : MonoBehaviour
{
    public Transform target;
    // Start is called before the first frame update
    void Start(){
        
    }

    // Update is called once per frame
    void Update(){
        this.transform.localPosition = this.target.localPosition;
        this.transform.localRotation = this.target.localRotation;
    }
}
