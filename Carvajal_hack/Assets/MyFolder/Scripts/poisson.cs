using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class poisson : MonoBehaviour
{


    // Update is called once per frame
    private void OnTriggerStay(Collider other) {
        if(other.name == "sheep"){
            Rigidbody rb = other.GetComponent<Rigidbody>();
            rb.AddForce(transform.up*20f);
            rb.AddForce(transform.forward*-10f);
        }
        
    }
}
