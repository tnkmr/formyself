/*! @brief NMF c/c++ header hpp
 * @author Tomohiko Nakamura 
 */
#ifndef NMF_HPP
#define NMF_HPP

#include <Eigen/Core>
/*! @namespace nmf
 * @brief namespace for NMF which includes NMF class.
 */
namespace nmf{
  const bool UNDEFINE_NMF_ALGORITHM_TYPE=false;
  /*! @brief NMF calculation class
   * @class NMF
   * We solve to find \f$ W,D \f$ such as \f$ Y\sim WD , W\geq0,D\geq0.\f$
   */
  class NMF{
  protected:
    //! distance or divergence type: 0=Euclied (default), 1=I-divergence, and 2=L1-Norm.
    int type;
  public:
    //! Weight Matrix
    Eigen::MatrixXd W;
    //! Dictionary Matrix
    Eigen::MatrixXd D;
    //! @brief constructor
    NMF(){type=0;}
    //! @brief destructor
    ~NMF(){;}
    /*! @brief set distance or divergence type
     * @param tp type number
     */
    void set(const int tp){type=tp;}
    /*! @brief update W
     * @param data data
     */
    void update_W(Eigen::MatrixXd *data);
    /*! @brief update D
     * @param data data
     */
    void update_D(Eigen::MatrixXd *data);
  };
};

void nmf::NMF::update_W(Eigen::MatrixXd *data){
  if(type==0)
    // L2-Norm
    W=W.cwiseProduct(((*data)*(D.transpose())).cwiseQuotient(W*D*(D.transpose())));
  else if(type==1){
    // KL-divergence
    W=W.cwiseProduct((((*data).cwiseQuotient(W*D))*(D.transpose())).cwiseQuotient(D.rowwise().sum().transpose().replicate(W.rows(),1)));
  }else assert(UNDEFINE_NMF_ALGORITHM_TYPE);
};

void nmf::NMF::update_D(Eigen::MatrixXd *data){
  if(type==0)
    // L2-Norm
    D=D.cwiseProduct((W.transpose()*(*data)).cwiseQuotient(W.transpose()*W*D));
  else if(type==1){
    // KL-divergence
    D=D.cwiseProduct(((W.transpose())*((*data).cwiseQuotient(W*D))).cwiseQuotient(W.colwise().sum().transpose().replicate(1,D.cols())));
  }else assert( UNDEFINE_NMF_ALGORITHM_TYPE);
};

#endif
