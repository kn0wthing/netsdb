#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "FFMatrixBlock.h"
#include "FFMatrixMeta.h"
#include "FFMatrixData.h"
#include "MHALayerNorm.h"
#include "MHAResConnection.h"
#include "FFMatrixBlockScanner.h"
#include "FFTransposeMult.h"
#include "FFRowAggregate.h"
#include "FFOutputLayer.h"

#include "FFMatrixPartitioner.h"
#include "FFReluBiasSum.h"
#include "FFTransposeBiasSum.h"
#include "SparseMatrixBlock.h"
#include "TensorBlock2D.h"

#include "FFMatrixWriter.h"

#include "FFInputLayerJoin.h"
#include "FFAggMatrix.h"

#include "FFMatrixBlock.h"
#include "FFMatrixUtil.h"
#include "SimpleFF.h"

#include "PDBClient.h"

using namespace std;

void print(pdb::PDBClient &pdbClient, string dbName, string setName)
{
  auto it = pdbClient.getSetIterator<FFMatrixBlock>(dbName, setName);

  for (auto r : it)
  {
    double *data = r->getRawDataHandle()->c_ptr();
    for (int i = 0; i < r->getRowNums() * r->getColNums(); i++)
    {
      std::cout << data[i] << ",";
    }
  }
}

void print_stats(pdb::PDBClient &pdbClient, string dbName, string setName)
{
  int rows = 0, cols = 0, blocks = 0;
  int totalRows = 0, totalCols = 0;
  int blockRows = 0, blockCols = 0;
  auto it = pdbClient.getSetIterator<FFMatrixBlock>(dbName, setName);

  for (auto r : it)
  {
    std::cout << r->getBlockRowIndex() << "," << r->getBlockColIndex() << ";";
    rows = r->getRowNums();
    cols = r->getColNums();
    if (r->getBlockRowIndex() == 0)
    {
      totalRows += r->getRowNums();
      blockRows += 1;
    }
    if (r->getBlockColIndex() == 0)
    {
      totalCols += r->getColNums();
      blockCols += 1;
    }
    blocks++;
  }

  std::cout << "\n"
            << setName << " (" << blockRows << " X " << blockCols << ") ("
            << blocks << ") : (" << totalRows << " x " << totalCols
            << "), Each block size: " << rows << " x " << cols << std::endl;
}

void inference_unit(pdb::PDBClient &pdbClient, string database, string w1,
                    string wo, string inputs, string b1, string bo,
                    string output, double dropout_rate) {
    string errMsg;

    {
        const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

        // make the computation
        pdb::Handle<pdb::Computation> readA =
            makeObject<FFMatrixBlockScanner>(database, w1);
        pdb::Handle<pdb::Computation> readB =
            makeObject<FFMatrixBlockScanner>(database, inputs);

        pdb::Handle<pdb::Computation> join = pdb::makeObject<FFTransposeMult>();
        join->setInput(0, readA);
        join->setInput(1, readB);

        // make the aggregation
        pdb::Handle<pdb::Computation> myAggregation =
            pdb::makeObject<FFAggMatrix>();
        myAggregation->setInput(join);

        pdb::Handle<pdb::Computation> readC =
            makeObject<FFMatrixBlockScanner>(database, b1);

        pdb::Handle<pdb::Computation> reluBias =
            pdb::makeObject<FFReluBiasSum>(dropout_rate);
        reluBias->setInput(0, myAggregation);
        reluBias->setInput(1, readC);

        // make the computation
        pdb::Handle<pdb::Computation> readD =
            makeObject<FFMatrixBlockScanner>(database, wo);

        pdb::Handle<pdb::Computation> join1 = pdb::makeObject<FFInputLayerJoin>();
        join1->setInput(0, readD);
        join1->setInput(1, reluBias);

        // make the aggregation
        pdb::Handle<pdb::Computation> myAggregation1 =
            pdb::makeObject<FFAggMatrix>();
        myAggregation1->setInput(join1);

        pdb::Handle<pdb::Computation> readE =
            makeObject<FFMatrixBlockScanner>(database, bo);

        pdb::Handle<pdb::Computation> reluBias1 =
            pdb::makeObject<FFTransposeBiasSum>();
        reluBias1->setInput(0, myAggregation1);
        reluBias1->setInput(1, readE);

        pdb::Handle<pdb::Computation> intermediateWriter =
            pdb::makeObject<FFMatrixWriter>(database, "yo");
        intermediateWriter->setInput(reluBias1);

        auto begin0 = std::chrono::high_resolution_clock::now();
        // run the computation
        if (!pdbClient.executeComputations(errMsg, "inference-unit-intermediate", false, intermediateWriter)) {
            cout << "Computation failed. Message was: " << errMsg << "\n";
            exit(1);
        }
        auto end0 = std::chrono::high_resolution_clock::now();
        std::cout << "Inference-unit Intermediate Stage Time Duration: "
                  << std::chrono::duration_cast<std::chrono::duration<float>>(end0 - begin0).count()
                  << " secs." << std::endl;

        pdb::Handle<pdb::Computation> readF =
            makeObject<FFMatrixBlockScanner>(database, "yo");

        pdb::Handle<pdb::Computation> expSum = pdb::makeObject<FFRowAggregate>();
        expSum->setInput(readF);

        pdb::Handle<pdb::Computation> softmax = pdb::makeObject<FFOutputLayer>();
        softmax->setInput(0, readF);
        softmax->setInput(1, expSum);

        // make the writer
        pdb::Handle<pdb::Computation> sumWriter =
            pdb::makeObject<FFMatrixWriter>(database, output);
        sumWriter->setInput(softmax);

        auto begin = std::chrono::high_resolution_clock::now();
        // run the computation
        if (!pdbClient.executeComputations(errMsg, "inference-unit", false, sumWriter)) {
            cout << "Computation failed. Message was: " << errMsg << "\n";
            exit(1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Inference-unit Stage Time Duration: "
                  << std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count()
                  << " secs." << std::endl;
    }
}

void loadMatrix(pdb::PDBClient &pdbClient, String dbName, String setName,
                int totalX, int totalY, int blockX, int blockY, int initVal,
                std::string &errMsg)
{

  int total = 0;
  pdb::makeObjectAllocatorBlock(128 * 1024 * 1024, true);

  pdb::Handle<pdb::Vector<pdb::Handle<FFMatrixBlock>>> storeMatrix1 =
      pdb::makeObject<pdb::Vector<pdb::Handle<FFMatrixBlock>>>();

  int numXBlocks = ceil(totalX / (double)blockX);
  int numYBlocks = ceil(totalY / (double)blockY);

  try
  {
    for (int i = 0; i < numXBlocks; i++)
    {
      for (int j = 0; j < numYBlocks; j++)
      {
        pdb::Handle<FFMatrixBlock> myData =
            pdb::makeObject<FFMatrixBlock>(i, j, blockX, blockY);

        for (int ii = 0; ii < blockX; ii++)
        {
          for (int jj = 0; jj < blockY; jj++)
          {
            // row = i * blockX + ii, col = j * blockY + jj
            double data =
                (i * blockX + ii) >= totalX || (j * blockY + jj) >= totalY
                    ? 0
                : initVal == -1 ? i + j + ii + jj
                                : initVal;
            (*(myData->getRawDataHandle()))[ii * blockY + jj] = data;
          }
        }

        // std::cout << "New block: " << total << std::endl;
        storeMatrix1->push_back(myData);
        total++;
      }
    }
    if (!pdbClient.sendData<FFMatrixBlock>(
            std::pair<std::string, std::string>(setName, dbName), storeMatrix1,
            errMsg))
    {
      std::cout << "Failed to send data to dispatcher server" << std::endl;
      exit(1);
    }
  }
  catch (NotEnoughSpace &e)
  {
    std::cout << "Failed to send data to dispatcher server" << std::endl;
    exit(1);
  }
  std::cout << setName << "(" << totalX << "x" << totalY << "): (" << numXBlocks
            << " x " << numYBlocks << ")" << total << " blocks = " << blockX
            << " x " << blockY << " each" << std::endl;

  // to write back all buffered records
  pdbClient.flushData(errMsg);
}

void loadLibrary(pdb::PDBClient &pdbClient, string path)
{
  string errMsg;
  if (!pdbClient.registerType(path, errMsg))
  {
    cout << "Couldnt include " << path << ": " << errMsg << endl;
    exit(-1);
  }
}

void createSet(pdb::PDBClient &pdbClient, string dbName, string setName,
               string setName1)
{
  string errMsg;
  if (!pdbClient.createSet<FFMatrixBlock>(
          dbName, setName, errMsg, (size_t)64 * (size_t)1024 * (size_t)1024,
          setName1))
  {
    cout << "Not able to create set: " + errMsg;
    exit(-1);
  }
  else
  {
    cout << "Created set.\n";
  }
}

void createDatabase(pdb::PDBClient &pdbClient, string dbName)
{
  string errMsg;
  if (!pdbClient.createDatabase(dbName, errMsg))
  {
    cout << "Not able to create database: " << errMsg << endl;
    exit(-1);
  }
  else
  {
    cout << "Created database" << endl;
  }
}

void create_weights_set(pdb::PDBClient & pdbClient, std::string weight_set_name, int numBlock_x, int block_x, int totalNumBlock_y,
	int block_y) {

     std::string errMsg;
     pdbClient.removeSet("mha", weight_set_name, errMsg);
     //create private set 
     pdbClient.createSet<FFMatrixBlock>("mha", weight_set_name, errMsg,
                     DEFAULT_PAGE_SIZE, weight_set_name, nullptr, nullptr, false);

     //load blocks to the private set 
     ff::loadMatrix(pdbClient, "mha", weight_set_name, numBlock_x, totalNumBlock_y, block_x, block_y, false, false, errMsg);

}

int main(int argc, char *argv[])
{
  string errMsg;

  string masterIp = "localhost";
  pdb::PDBLoggerPtr clientLogger = make_shared<pdb::PDBLogger>("MHAclientLog");
  pdb::PDBClient pdbClient(8108, masterIp, clientLogger, false, true);
  pdb::CatalogClient catalogClient(8108, masterIp, clientLogger);

  loadLibrary(pdbClient, "libraries/libFFMatrixMeta.so");
  loadLibrary(pdbClient, "libraries/libFFMatrixData.so");
  loadLibrary(pdbClient, "libraries/libFFMatrixBlock.so");

  loadLibrary(pdbClient, "libraries/libFFMatrixBlockScanner.so");
  loadLibrary(pdbClient, "libraries/libFFMatrixWriter.so");

  loadLibrary(pdbClient, "libraries/libFFRowAggregate.so");
  loadLibrary(pdbClient, "libraries/libFFOutputLayer.so");
  loadLibrary(pdbClient, "libraries/libFFTransposeMult.so");
  loadLibrary(pdbClient, "libraries/libFFInputLayerJoin.so");
  loadLibrary(pdbClient, "libraries/libFFAggMatrix.so");

// "FFMatrixPartitioner.h"
// #include "FFReluBiasSum.h"
// #include "FFTransposeBiasSum.h"
// #include "SparseMatrixBlock.h"
// #include "TensorBlock2D.h"
  loadLibrary(pdbClient, "libraries/libFFMatrixPartitioner.so");
  loadLibrary(pdbClient, "libraries/libFFReluBiasSum.so");
  loadLibrary(pdbClient, "libraries/libFFTransposeBiasSum.so");

  loadLibrary(pdbClient, "libraries/libMHALayerNorm.so");
  loadLibrary(pdbClient, "libraries/libMHAResConnection.so");


  createDatabase(pdbClient, "mha");

  createSet(pdbClient, "mha", "input", "Input");

  createSet(pdbClient, "mha", "w_k", "WKey");
  createSet(pdbClient, "mha", "w_q", "WQuery");
  createSet(pdbClient, "mha", "w_v", "WValue");

  createSet(pdbClient, "mha", "b_k", "WForget");
  createSet(pdbClient, "mha", "b_q", "WInput");
  createSet(pdbClient, "mha", "b_v", "WOutput");

  // ff:: createDatabase(pdbClient, "ff");
  // ff:: setup(pdbClient, "ff");

  // if not working add ff::
  ff::createSet(pdbClient, "mha", "inputs", "inputs", 64);
  ff::createSet(pdbClient, "mha", "label", "label", 64);

  ff::createSet(pdbClient, "mha", "w1", "W1", 64);
  ff::createSet(pdbClient, "mha", "b1", "B1", 64);

  ff::createSet(pdbClient, "mha", "w0", "W0", 64);
  ff::createSet(pdbClient, "mha", "b0", "B0", 64);

  ff::createSet(pdbClient, "mha", "output", "Output", 256);
  ff::createSet(pdbClient, "mha", "block_output", "Block_Output", 256);

  int context_size = 10; // features
  int B = 1;             // batch size
  int em_size = 64;      // output labels?
  int block_x = 16;
  int block_y = 16;
  // Feed forward q,k,v to w and b

  // loadMatrix(pdbClient, "mha", "input", context_size, em_size, block_x, block_y, 2, errMsg);
  // loadMatrix(pdbClient, "mha", "w_k", em_size, em_size, block_x, block_y, 1, errMsg);
  // loadMatrix(pdbClient, "mha", "w_q", em_size, em_size, block_x, block_y, 1, errMsg);
  // loadMatrix(pdbClient, "mha", "w_v", em_size, em_size, block_x, block_y, 1, errMsg);

  create_weights_set(pdbClient, "input", context_size, em_size, block_x, block_y);
  create_weights_set(pdbClient, "w_k", em_size, em_size, block_x, block_y);
  create_weights_set(pdbClient, "w_q", em_size, em_size, block_x, block_y);
  create_weights_set(pdbClient, "w_v", em_size, em_size, block_x, block_y);

  // context size  = batch size and numFeatures = em_size
  ff::loadMatrix(pdbClient, "mha", "w0", 16, em_size, block_x, block_y, false, false, errMsg);
  ff::loadMatrix(pdbClient, "mha", "w1", context_size, 16, block_x, block_y, false, false, errMsg);
  ff::loadMatrix(pdbClient, "mha", "b0", 16, 1, block_x, 1, false, true, errMsg);
  ff::loadMatrix(pdbClient, "mha", "b1", context_size, 1, block_x, 1, false, true, errMsg);
  std::cout << "-----------load matrix---------------: " << "\n";
  double dropout_rate = 0.5;
  {
    const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
    
    std::cout << "-----------Temp Block---------------: " << "\n";

    // ----------------------------------------------
    // make the computation
    pdb::Handle<pdb::Computation> readA =
        makeObject<FFMatrixBlockScanner>("mha", "input");
    pdb::Handle<pdb::Computation> readB =
        makeObject<FFMatrixBlockScanner>("mha", "w_k");

    pdb::Handle<pdb::Computation> key_matrix = pdb::makeObject<FFTransposeMult>();
    key_matrix->setInput(0, readA);
    key_matrix->setInput(1, readB);

    pdb::Handle<pdb::Computation> readC =
        makeObject<FFMatrixBlockScanner>("mha", "w_q");
    pdb::Handle<pdb::Computation> query_matrix = pdb::makeObject<FFTransposeMult>();
    query_matrix->setInput(0, readA);
    query_matrix->setInput(1, readC);

    pdb::Handle<pdb::Computation> readD =
        makeObject<FFMatrixBlockScanner>("mha", "w_v");
    pdb::Handle<pdb::Computation> value_matrix = pdb::makeObject<FFTransposeMult>();
    value_matrix->setInput(0, readA);
    value_matrix->setInput(1, readB);

    pdb::Handle<pdb::Computation> attention1 = pdb::makeObject<FFTransposeMult>();
    attention1->setInput(0, query_matrix);
    attention1->setInput(1, key_matrix);

    pdb::Handle<pdb::Computation> intermediateWriter =
        pdb::makeObject<FFMatrixWriter>("mha", "i-0");
    intermediateWriter->setInput(attention1);

    pdb::Handle<pdb::Computation> readF =
        makeObject<FFMatrixBlockScanner>("mha", "i-0");

    pdb::Handle<pdb::Computation> expSum = pdb::makeObject<FFRowAggregate>();
    expSum->setInput(readF);

    pdb::Handle<pdb::Computation> softmax = pdb::makeObject<FFOutputLayer>();
    softmax->setInput(0, readF);
    softmax->setInput(1, expSum);

    pdb::Handle<pdb::Computation> attention = pdb::makeObject<FFTransposeMult>();
    attention->setInput(0, softmax);
    attention->setInput(1, value_matrix);
    
    std::cout << "-----------After attention---------------: " << "\n";

    pdb::Handle<pdb::Computation> intermediateWriter1 =
        pdb::makeObject<FFMatrixWriter>("mha", "ou");
    intermediateWriter1->setInput(attention);

    std::cout << "-----------Intermediate Writer1---------------: " << "\n";

    inference_unit(pdbClient, "mha", "w1", "w0", "ou", "b1", "b0",
                       "output", dropout_rate);

    std::cout << "-----------After inference unit---------------: " << "\n";

    pdb::Handle<pdb::Computation> readG =
        makeObject<FFMatrixBlockScanner>("mha", "output");
    pdb::Handle<pdb::Computation> layernormalization = pdb::makeObject<MHALayerNorm>();
    layernormalization->setInput(0, readG);
    
    std::cout << "-----------After LayerNorm---------------: " << "\n";
    

    pdb::Handle<pdb::Computation> rescon = pdb::makeObject<MHAResConnection>();
    rescon->setInput(0, readA);
    rescon->setInput(1, layernormalization);

    std::cout << "-----------After res connection---------------: " << "\n";

    // make the writer
    pdb::Handle<pdb::Computation> myWriter = pdb::makeObject<FFMatrixWriter>("mha", "block_output");
    myWriter->setInput(rescon);

    // run the computation
    if (!pdbClient.executeComputations(errMsg, myWriter))
    {
      std::cout << "Computation failed. Message was: " << errMsg << "\n";
      return 1;
    }
  }

  {
    const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // print_stats(pdbClient, "mha", "ou");
    // print_stats(pdbClient, "mha", "input");
    // print_stats(pdbClient, "mha", "output");
    // print_stats(pdbClient, "mha", "w_k");
    // print_stats(pdbClient, "mha", "w_q");
    // print_stats(pdbClient, "mha", "w_v");
  }

  return 0;
}