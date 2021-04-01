using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereCoordsTest : MonoBehaviour{
    // Start is called before the first frame update
    public float radius, theta, phi, enemy_r, enemy_theta;
    public Transform center, enemy;
    private Vector3 initial_pos;
    void Start(){        
        // this.radius = this.spawn_radius;
        // phi 0, 180 or -45, 225
        // theta = 0 180/90 with arm
        this.initial_pos = this.center.position;
    }

    public void DrawVector(Vector3 starting_point, Vector3 dir, Color color){
        Debug.DrawLine(starting_point, starting_point + dir, color);
    }
    public Vector3 p0, n, u, v;
    public float spd, min_targ_dist=0.12f, min_agent_dist, interp;
    // Update is called once per frame
    void Update(){
        float phir, thetar, tx, ty, tz, ex, ey;
        thetar = Mathf.Clamp(theta * Mathf.Deg2Rad, -Mathf.PI, Mathf.PI);
        phir = Mathf.Clamp(phi * Mathf.Deg2Rad, -2f * Mathf.PI, 2f * Mathf.PI);
        Vector3 final_pos, initial_pos;
        // Move the target sphere to a new spot
        tx = radius * Mathf.Sin(thetar) * Mathf.Cos(phir);
        ty = radius * Mathf.Sin(thetar) * Mathf.Sin(phir);
        tz = radius * Mathf.Cos(thetar);
        this.transform.position = new Vector3(tx, ty, tz) + this.initial_pos;
        enemy_theta += Time.deltaTime * spd;
        interp += Time.deltaTime * spd;
        if(enemy_theta > 360.0f)
            enemy_theta = 0;
        if(interp > 1.0f){
            interp = 1.0f;
            spd *= -1;
        }
        else if(interp < 0){
            interp = 0.0f;
            spd *= -1;
        }            
        initial_pos = this.initial_pos + min_agent_dist*(this.transform.position - this.initial_pos).normalized;
        final_pos = this.transform.position - min_targ_dist*(this.transform.position - this.initial_pos).normalized;
        p0 = initial_pos + (final_pos - initial_pos)* interp;
        n = (this.transform.position - this.initial_pos).normalized;
        Debug.Log(Vector3.Distance(p0, this.initial_pos));
        // U and V need to be orthogonal to n.
        if(( !(Mathf.Abs(n.z) < 1e-5) || !(Mathf.Abs(n.x) < 1e-5)) )
            v = new Vector3(-n.z, 0, n.x).normalized;
        else
            v = new Vector3(-n.y, n.x, 0).normalized;
        
        u = Vector3.Cross(n, v).normalized;

        ex = enemy_r * Mathf.Cos(enemy_theta * Mathf.Deg2Rad);
        ey = enemy_r * Mathf.Sin(enemy_theta * Mathf.Deg2Rad);
        this.enemy.position = p0 + ex*v + ey*u;        
    }

    // void OnDrawGizmos(){        
    //     Gizmos.color = Color.yellow;          
    //     Gizmos.DrawLine(center.position, this.transform.position); 
    //     Gizmos.color = Color.green;            
    //     Gizmos.DrawLine(p0, p0 + enemy_r*v); 
    //     Gizmos.color = Color.red;            
    //     Gizmos.DrawLine(p0, this.enemy.position); 
    //     Gizmos.color = Color.cyan;            
    //     Gizmos.DrawSphere(p0, 0.01f);
    //     // Gizmos.color = Color.red;            
    //     // Gizmos.DrawLine(p0, p0 + v*0.25f); 
    //     // Gizmos.color = Color.blue;            
    //     // Gizmos.DrawLine(p0, p0 + u*0.25f); 
    // }
}
