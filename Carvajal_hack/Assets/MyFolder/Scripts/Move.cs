using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Move : MonoBehaviour

{
  
    Animator anim;
    Rigidbody rb;
    public float x;
    void Start()
    {
        anim = GetComponent<Animator>();
        rb = GetComponent<Rigidbody>();
    }
private void OnTriggerEnter(Collider other) {
    if(other.name == "adelante"){
            Vector3 movement = new Vector3(x, 0.0f, 0.0f);
        
    }
}
}
