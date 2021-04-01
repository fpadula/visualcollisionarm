using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ManipulatorJoint : MonoBehaviour{    
    // Flag que avisa se a junta está na posição alvo
    public bool reached_target_angle, update_angle;
    // Propriedades da junta (esforço, velocidade, ângulo etc)
    public float effort, velocity, angle;
    // ângulo alvo, erro angular e tolerância (para considerar que a junta chegou a posição alvo)
    public float target_angle, desired_angle, error, tol, min_angle_diff_to_change;
    // Variável que armazena referencia para a HingeJoint    
    private HingeJoint hj;    
    private Rigidbody rb;
    private float cw_dir, ccw_dir, intermediate_angle;
    private bool has_intermediate_angle;

    private Vector2 Angle2Vec(float angle){
        return new Vector2(Mathf.Cos(Mathf.Deg2Rad*angle), Mathf.Sin(Mathf.Deg2Rad*angle));
    }

    private float MinAngleDiff(float angleA, float angleB){
        angleA = mod(angleA, 360.0f);
        angleB = mod(angleB, 360.0f);
        if(angleA < angleB)
            return angleB - angleA;
        return angleA - angleB;
    }

    private float CartesianAngleSqrdDiff(float angleA, float angleB){
        Vector2 Av, Bv;
        Av = Angle2Vec(angleA);
        Bv = Angle2Vec(angleB);
        return (Av.x - Bv.x)*(Av.x - Bv.x) + (Av.y - Bv.y)*(Av.y - Bv.y);
    }

    private bool InInterval(float from_a, float to_a, float angle){
        if(to_a < from_a)
            // this means that the interval crosses the 0 position, so we have to split the checking
            // return InInterval(from_a, 360.0f, angle) || InInterval(0, to_a, angle);
            return (angle >= from_a) || (angle <= to_a);
        else
            return (angle >= from_a) && (angle <= to_a);
    }

    private bool SmallestViolatesLimits(float target_angle, float current_angle){
        float cw_delta,ccw_delta, to_a, from_a;
        bool ret;

        ret = false;
        if(this.hj.useLimits){
            target_angle = mod(target_angle, 360.0f);
            current_angle = mod(current_angle, 360.0f);
            // Debug.Log(target_angle + ", " + current_angle);
            cw_delta = mod((current_angle - target_angle), 360.0f);
            ccw_delta = mod((target_angle - current_angle), 360.0f);     
            if(cw_delta < ccw_delta){
                to_a = current_angle;
                from_a = target_angle;                
            }
            else{
                to_a = target_angle;
                from_a = current_angle;                
            }   
            if(InInterval(from_a, to_a, mod(this.hj.limits.min, 360.0f)) || InInterval(from_a, to_a, mod(this.hj.limits.max, 360.0f))){
                // Violated limits, reversing direction of turn                
                ret = true;
            }
        }
        return ret;
    }

    private float SmallestDirection(float target_angle, float current_angle){
        float cw_delta,ccw_delta, error, to_a, from_a;        

        target_angle = mod(target_angle, 360.0f);
        current_angle = mod(target_angle, 360.0f);
        cw_delta = mod((current_angle - target_angle), 360.0f);
        ccw_delta = mod((target_angle - current_angle), 360.0f);
        if(cw_delta < ccw_delta){
            return this.cw_dir;
        }
        else{
            return this.ccw_dir;
        }
    }

    public float GetMinLim(){
        if(this.hj.useLimits)
            return this.hj.limits.min;
        else
            return -180.0f;
    }

    public float GetMaxLim(){
        if(this.hj.useLimits)
            return this.hj.limits.max;
        else
            return 180.0f;
    }

    private float GetError(float target_angle, float current_angle){
        float cw_delta,ccw_delta, error, to_a, from_a;        
        cw_delta = mod((current_angle - target_angle), 360.0f);
        ccw_delta = mod((target_angle - current_angle), 360.0f);
        if(cw_delta < ccw_delta){
            to_a = current_angle;
            from_a = target_angle;
            error = cw_delta * this.cw_dir;
        }
        else{
            to_a = target_angle;
            from_a = current_angle;
            error = ccw_delta * this.ccw_dir;
        }
        if(this.hj.useLimits){
            if(InInterval(from_a, to_a, mod(this.hj.limits.min, 360.0f)) || InInterval(from_a, to_a, mod(this.hj.limits.max, 360.0f))){
                // Violated limits, reversing direction of turn                
                if(cw_delta < ccw_delta){
                    error = ccw_delta * this.ccw_dir;
                }
                else{
                    error = cw_delta * this.cw_dir;
                }
            }
        }
        return error;
    }

    // Start é chamado antes do primeiro frame
    void Start(){
        this.hj = GetComponent<HingeJoint>();
        this.rb = GetComponent<Rigidbody>();
        this.error = 0;        
        this.cw_dir = 1.0f;
        this.ccw_dir = -1.0f;
        this.has_intermediate_angle = false;
    }

    // FixedUpdate é chamado em cada passo da engine física
    void FixedUpdate(){
        if(this.update_angle){
            this.update_angle = false;
            SetJointAngle(desired_angle);
        }
        this.angle = this.hj.angle;
        if(this.has_intermediate_angle){
            if(!SmallestViolatesLimits(this.intermediate_angle, this.angle)){
                SetJointAngle(this.intermediate_angle);
                this.has_intermediate_angle = false;
            }
        }
        // Extrai as informações pertinentes da junta        
        this.error = CartesianAngleSqrdDiff(this.angle, this.target_angle);
        
        this.velocity = this.hj.velocity;
        // Se o erro for menor que tol, então a junta chegou na posição alvo
        this.reached_target_angle = (this.error <= this.tol);
    }

    float mod(float x, float m) {
        return (x%m + m)%m;
    }

    public void SetJointAngle(float angle){
        // this.target_angle = mod(angle, 360.0f);        
        if(CartesianAngleSqrdDiff(angle, this.angle) > this.min_angle_diff_to_change){
            if(SmallestViolatesLimits(angle, this.angle)){
                // Debug.Log("Violated limits!");
                // Debug.Log("Error: " + GetError(angle, this.angle));
                this.has_intermediate_angle = true;
                this.intermediate_angle = angle;
                angle += GetError(angle, this.angle)/2.0f;
            }
            // else
            //     Debug.Log("Didnt violate limits!");
            this.target_angle = angle;
            JointSpring curr_spring;
            curr_spring = this.hj.spring;
            curr_spring.targetPosition = angle;
            this.hj.spring = curr_spring;
            this.reached_target_angle = false;
        }
    }

    public void SetLimit(bool enabled){ this.hj.useLimits = enabled; }

    // Retorna verdadeiro caso a junta tenha chegado à posição alvo
    public bool JointReachedTargetPosition(){return this.reached_target_angle;}
}
