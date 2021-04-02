using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class RegisterStringLogSideChannel : MonoBehaviour
{

    StringLogSideChannel stringChannel;
    public void Awake()
    {
        // We create the Side Channel
        stringChannel = new StringLogSideChannel();

        // When a Debug.Log message is created, we send it to the stringChannel
        // Application.logMessageReceived += stringChannel.SendDebugStatementToPython;

        // The channel must be registered with the SideChannelsManager class
        SideChannelsManager.RegisterSideChannel(stringChannel);
    }

    public void SendString(string message){
        stringChannel.SendString(message);
    }

    // public void OnDestroy()
    // {
    //     // De-register the Debug.Log callback
    //     Application.logMessageReceived -= stringChannel.SendDebugStatementToPython;
    //     if (Academy.IsInitialized){
    //         SideChannelsManager.UnregisterSideChannel(stringChannel);
    //     }
    // }
        
}