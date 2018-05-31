// pcl_conversions.h does weird things to other packages when it is included for
//   compilation. It causes compiler to give errors that are not errors.
// So copied just the stuff I need here, to my own package, to see if it works
//   better.

#ifndef PCL_CONVERSIONS_H__
#define PCL_CONVERSIONS_H__

#include <ros/ros.h>

#include <pcl/PCLHeader.h>


namespace pcl_conversions {

  /** PCLHeader <=> Header **/

  void fromPCL(const ::pcl::PCLHeader &pcl_header, ::std_msgs::Header &header)
  {
    header.stamp.fromNSec(pcl_header.stamp * 1000ull);  // Convert from us to ns
    header.seq = pcl_header.seq;
    header.frame_id = pcl_header.frame_id;
  }

  void toPCL(const std_msgs::Header &header, pcl::PCLHeader &pcl_header)
  {
    pcl_header.stamp = header.stamp.toNSec() / 1000ull;  // Convert from ns to us
    pcl_header.seq = header.seq;
    pcl_header.frame_id = header.frame_id;
  }

  std_msgs::Header fromPCL(const pcl::PCLHeader &pcl_header)
  {
    std_msgs::Header header;
    fromPCL(pcl_header, header);
    return header;
  }


  /** PCLPointField <=> PointField **/

  void fromPCL(const pcl::PCLPointField &pcl_pf, sensor_msgs::PointField &pf)
  {
    pf.name = pcl_pf.name;
    pf.offset = pcl_pf.offset;
    pf.datatype = pcl_pf.datatype;
    pf.count = pcl_pf.count;
  }

  void fromPCL(const std::vector<pcl::PCLPointField> &pcl_pfs, std::vector<sensor_msgs::PointField> &pfs)
  {
    pfs.resize(pcl_pfs.size());
    std::vector<pcl::PCLPointField>::const_iterator it = pcl_pfs.begin();
    int i = 0;
    for(; it != pcl_pfs.end(); ++it, ++i) {
      fromPCL(*(it), pfs[i]);
    }
  }

  void toPCL(const sensor_msgs::PointField &pf, pcl::PCLPointField &pcl_pf)
  {
    pcl_pf.name = pf.name;
    pcl_pf.offset = pf.offset;
    pcl_pf.datatype = pf.datatype;
    pcl_pf.count = pf.count;
  }

  void toPCL(const std::vector<sensor_msgs::PointField> &pfs, std::vector<pcl::PCLPointField> &pcl_pfs)
  {
    pcl_pfs.resize(pfs.size());
    std::vector<sensor_msgs::PointField>::const_iterator it = pfs.begin();
    int i = 0;
    for(; it != pfs.end(); ++it, ++i) {
      toPCL(*(it), pcl_pfs[i]);
    }
  }


} // namespace pcl_conversions


namespace ros
{
  template<>
  struct DefaultMessageCreator<pcl::PCLPointCloud2>
  {
    boost::shared_ptr<pcl::PCLPointCloud2> operator() ()
    {
      boost::shared_ptr<pcl::PCLPointCloud2> msg(new pcl::PCLPointCloud2());
      return msg;
    }
  };

  namespace message_traits
  {
    template<>
    struct MD5Sum<pcl::PCLPointCloud2>
    {
      static const char* value() { return MD5Sum<sensor_msgs::PointCloud2>::value(); }
      static const char* value(const pcl::PCLPointCloud2&) { return value(); }

      static const uint64_t static_value1 = MD5Sum<sensor_msgs::PointCloud2>::static_value1;
      static const uint64_t static_value2 = MD5Sum<sensor_msgs::PointCloud2>::static_value2;

      // If the definition of sensor_msgs/PointCloud2 changes, we'll get a compile error here.
      ROS_STATIC_ASSERT(static_value1 == 0x1158d486dd51d683ULL);
      ROS_STATIC_ASSERT(static_value2 == 0xce2f1be655c3c181ULL);
    };

    template<>
    struct DataType<pcl::PCLPointCloud2>
    {
      static const char* value() { return DataType<sensor_msgs::PointCloud2>::value(); }
      static const char* value(const pcl::PCLPointCloud2&) { return value(); }
    };

    template<>
    struct Definition<pcl::PCLPointCloud2>
    {
      static const char* value() { return Definition<sensor_msgs::PointCloud2>::value(); }
      static const char* value(const pcl::PCLPointCloud2&) { return value(); }
    };

    template<> struct HasHeader<pcl::PCLPointCloud2> : TrueType {};
  } // namespace ros::message_traits

  namespace serialization
  {
    /*
     * Provide a custom serialization for pcl::PCLPointCloud2
     */
    template<>
    struct Serializer<pcl::PCLPointCloud2>
    {
      template<typename Stream>
      //inline
      static void write(Stream& stream, const pcl::PCLPointCloud2& m)
      {
        std_msgs::Header header;
        pcl_conversions::fromPCL(m.header, header);
        stream.next(header);
        stream.next(m.height);
        stream.next(m.width);
        std::vector<sensor_msgs::PointField> pfs;
        pcl_conversions::fromPCL(m.fields, pfs);
        stream.next(pfs);
        stream.next(m.is_bigendian);
        stream.next(m.point_step);
        stream.next(m.row_step);
        stream.next(m.data);
        stream.next(m.is_dense);
      }

      template<typename Stream>
      //inline
      static void read(Stream& stream, pcl::PCLPointCloud2& m)
      {
        std_msgs::Header header;
        stream.next(header);
        pcl_conversions::toPCL(header, m.header);
        stream.next(m.height);
        stream.next(m.width);
        std::vector<sensor_msgs::PointField> pfs;
        stream.next(pfs);
        pcl_conversions::toPCL(pfs, m.fields);
        stream.next(m.is_bigendian);
        stream.next(m.point_step);
        stream.next(m.row_step);
        stream.next(m.data);
        stream.next(m.is_dense);
      }

      //inline
      static uint32_t serializedLength(const pcl::PCLPointCloud2& m)
      {
        uint32_t length = 0;

        std_msgs::Header header;
        pcl_conversions::fromPCL(m.header, header);
        length += serializationLength(header);
        length += 4; // height
        length += 4; // width
        std::vector<sensor_msgs::PointField> pfs;
        pcl_conversions::fromPCL(m.fields, pfs);
        length += serializationLength(pfs); // fields
        length += 1; // is_bigendian
        length += 4; // point_step
        length += 4; // row_step
        length += 4; // data's size
        length += m.data.size() * sizeof(pcl::uint8_t);
        length += 1; // is_dense

        return length;
      }
    };

    /*
     * Provide a custom serialization for pcl::PCLPointField
     */
    template<>
    struct Serializer<pcl::PCLPointField>
    {
      template<typename Stream>
      //inline
      static void write(Stream& stream, const pcl::PCLPointField& m)
      {
        stream.next(m.name);
        stream.next(m.offset);
        stream.next(m.datatype);
        stream.next(m.count);
      }

      template<typename Stream>
      //inline
      static void read(Stream& stream, pcl::PCLPointField& m)
      {
        stream.next(m.name);
        stream.next(m.offset);
        stream.next(m.datatype);
        stream.next(m.count);
      }

      //inline
      static uint32_t serializedLength(const pcl::PCLPointField& m)
      {
        uint32_t length = 0;

        length += serializationLength(m.name);
        length += serializationLength(m.offset);
        length += serializationLength(m.datatype);
        length += serializationLength(m.count);

        return length;
      }
    };

    /*
     * Provide a custom serialization for pcl::PCLHeader
     */
    template<>
    struct Serializer<pcl::PCLHeader>
    {
      template<typename Stream>
      //inline
      static void write(Stream& stream, const pcl::PCLHeader& m)
      {
        std_msgs::Header header;
        pcl_conversions::fromPCL(m, header);
        stream.next(header);
      }

      template<typename Stream>
      //inline
      static void read(Stream& stream, pcl::PCLHeader& m)
      {
        std_msgs::Header header;
        stream.next(header);
        pcl_conversions::toPCL(header, m);
      }

      //inline
      static uint32_t serializedLength(const pcl::PCLHeader& m)
      {
        uint32_t length = 0;

        std_msgs::Header header;
        pcl_conversions::fromPCL(m, header);
        length += serializationLength(header);

        return length;
      }
    };
  } // namespace ros::serialization

} // namespace ros



#endif /* PCL_CONVERSIONS_H__ */
