#pragma once
#include"NNNode.h"
#include"NNPacket.h"
#include"NNAddress.h"

enum class NNNodeState : uint8_t {
    WAITING_FOR_INPUT,
    READY_TO_EXECUTE,
    EXECUTING,
    WAITING_FOR_TURN,
    READY_TO_TRANSMIT,
    TRANSMITTED
};

struct NNWindowConfig {
    uint8_t  ownLayerId;              // this node's own layer, to filter sibling observations
    uint16_t precedingSiblingsMask;   // which sibling nodeIds (within this layer) must transmit before this node
};

class NNScheduler {
public:
    NNScheduler(NNNode& nodeRef, const NNWindowConfig& cfg)
        : node(nodeRef), windowCfg(cfg), state(NNNodeState::WAITING_FOR_INPUT), observedSiblingsMask(0) {}

    // Called for EVERY packet seen on the medium (not just this node's own inputs). lets the scheduler track sibling turn-taking independently of NNNode's own input logic.
    void onPacketObserved(const NNPacket& pkt) {
        NNAddress src = decodeAddress(pkt.header.sourceAddress);
        if (src.layerId == windowCfg.ownLayerId) {
            observedSiblingsMask |= (uint16_t(1) << src.nodeId);
        }
    }

    // Advances the state machine by one step per call. Intended to be called once per iteration.
    void tick() {
        switch (state) {
            case NNNodeState::WAITING_FOR_INPUT:
                if (node.readyToExecute()) state = NNNodeState::READY_TO_EXECUTE;
                break;
            case NNNodeState::READY_TO_EXECUTE:
                state = NNNodeState::EXECUTING;
                break;
            case NNNodeState::EXECUTING:
                node.execute();
                state = NNNodeState::WAITING_FOR_TURN;
                break;
            case NNNodeState::WAITING_FOR_TURN:
                if ((observedSiblingsMask & windowCfg.precedingSiblingsMask) == windowCfg.precedingSiblingsMask) {
                    state = NNNodeState::READY_TO_TRANSMIT;
                }
                break;
            case NNNodeState::READY_TO_TRANSMIT:
            case NNNodeState::TRANSMITTED:
                break; 
        }
    }

    // Returns true and fills outPkt once when it's this node's turn.
    bool hasOutputReady(NNPacket& outPkt) {
        if (state != NNNodeState::READY_TO_TRANSMIT) return false;
        outPkt = node.buildOutputPacket();
        state = NNNodeState::TRANSMITTED;
        return true;
    }

    NNNodeState getState() const { return state; }

    void resetForNextPass() {
        node.resetForNextPass();
        state = NNNodeState::WAITING_FOR_INPUT;
        observedSiblingsMask = 0;
    }

private:
    NNNode& node;
    NNWindowConfig windowCfg;
    NNNodeState state;
    uint16_t observedSiblingsMask;
};