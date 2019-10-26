using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class nube : MonoBehaviour
{
public float x;
public float limite;

private void Update() {
    if(this.transform.position.x < limite){
        this.transform.position = new Vector3(x,this.transform.position.y,this.transform.position.z);
    }else{
        this.transform.position = new Vector3(this.transform.position.x-.2f,this.transform.position.y,this.transform.position.z);
    }
}

    // Update is called once per frame
 
    }

