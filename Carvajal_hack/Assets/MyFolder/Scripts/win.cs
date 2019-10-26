using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class win : MonoBehaviour
{

[SerializeField] public GameObject sucess;
//public GameObject boton;
        void Start()
    {
        sucess.SetActive(false);
    }





    // Start is called before the first frame update
 private void OnCollisionEnter(Collision other) {
    if(other.gameObject.tag == "baby"){
           sucess.SetActive(true);
        //   delay(3000);
        

    }
    }
}
